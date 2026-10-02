import { atom, read, update } from 'claude-code'
import type { EngineInterface, PluginOptions, Register } from 'claude-code'

import type { Live, LiveMode, LiveState } from '../types'

const PLUGIN = 'live-vibe'
const OFF: Live = { isOn: false, mode: 'live', state: 'off', port: 0, turnId: null, vibeBefore: false, gen: 0 }
const live = atom({ plugin: 'live-vibe', key: 'live' } as const, OFF)
const isVibe = atom({ plugin: 'live-vibe', key: 'isVibe' } as const, false)

// The director reads and directs; every other tool is a worker's job.
const DIRECTOR_TOOLS = new Set([
  'Read', 'Agent', 'SendMessage', 'ListAgents', 'TaskStop',
  'TaskCreate', 'TaskUpdate', 'TaskList', 'ToolSearch', 'AskUserQuestion',
])

// After Oh My Pi's vibe-mode-active.md, with its vibe_* tools mapped onto Claude Code's own.
const VIBE_SECTION = `<vibe-mode>
Vibe mode ON. You are DIRECTOR: drive worker subagents, full coding agents with every normal tool; NEVER edit, run, grep, or build yourself. Verify work by reading files.

Toolset: Read, Agent (spawn a worker, run_in_background: true), SendMessage (continue a worker by name), ListAgents, TaskStop (kill a worker), TaskCreate/TaskUpdate/TaskList (your own bookkeeping), AskUserQuestion.

# Workers

- fast: model "sonnet"; mechanical, well-specified work: renames, small fixes, boilerplate, data collection, tests and output reports.
- good: model "opus"; design, tricky debugging, multi-file refactors, judgment-heavy work.

Workers are persistent conversations: give each a name, one worker per workstream, keep it on that workstream. Spawn once, then SendMessage the SAME worker for follow-ups; NEVER respawn it.

# Direction

1. Split requests into independent workstreams.
2. Spawn each with a complete self-contained brief: files, constraints, acceptance criteria. Workers start blank; they never see this conversation.
3. Spawns and sends return immediately; a worker's result arrives as a task notification when its turn finishes. Direct other workers meanwhile; never poll.
4. On each result, Read the touched files to verify claims before building on them; then SendMessage corrections, the next step, or a review request. Track parent tasks with TaskCreate/TaskUpdate; workers do not own this bookkeeping.
5. Route by difficulty: draft with fast; escalate to good if fast stalls or judgment is needed. good designs; fast executes mechanical parts.
6. TaskStop stuck workers or workers whose workstream is done; ListAgents if the roster is lost.

Run workers concurrently, normally one fast and one good on different workstreams. The final outcome is yours: verify with Read; do not take a worker's word for it.
</vibe-mode>`

const VOICE_SECTION = `<live-voice>
Live voice is ON. The user is speaking, and your final answer of each turn is read aloud by a speech synthesizer; the transcript is still on screen. Do the work with tools as usual, but make the final visible answer one to three short spoken sentences: no markdown, no lists, no code blocks, no URLs. Say file names and numbers plainly. Anything long belongs in a file the user can open, named in one sentence. Bracketed notes in the conversation such as "The user spoke over you" or "they heard only" are system annotations, not your words: the first is a live interruption to act on at once, the second says how much of an answer was heard, so do not repeat it.
</live-voice>`

// In live vibe the user talks to a small voice model; Claude's answers reach them only through its summary.
const RELAY_SECTION = `<live-vibe-relay>
Live vibe is ON. The user is talking to a voice front, a small fast model that hands their spoken requests to you and relays your final answer of each turn back to them, summarized in one or two spoken sentences. End every turn with a concise plain-text report the front can summarize: what you did, what you found or changed, whether it is verified, and whether work is still running in the background (workers spawned, results pending). No spoken style needed, but lead with the outcome and leave out long listings unless asked. A bracketed note such as "[The user, by voice, adds ...]" is a live addition to act on at once.
</live-vibe-relay>`

// A whole utterance that only asks for another model ("switch to sonnet", "change the model to opus please", "use haiku").
// Anchored at both ends, so a request that merely mentions a model ("use opus to review this") stays a prompt.
// The live vibe sidecar receives this source as --switch-pattern, so both modes match the same words.
const SPOKEN_SWITCH = /^(?:(?:ok|okay|hey|alright) )?(?:claude )?(?:please )?(?:(?:switch|change|swap|set)(?: over)?(?: the)?(?: model)? to|use)(?: the)? (opus|sonnet|haiku|fable)(?: model)?(?: please)?$/

function spokenModel(text: string): string | undefined {
  return SPOKEN_SWITCH.exec(text.toLowerCase().replace(/[^a-z ]+/g, ' ').replace(/\s+/g, ' ').trim())?.[1]
}

// Where Claude's words go: read aloud as they are (/live), or through the front's announcer (/livevibe).
function replyPath(l: Live) {
  return l.mode === 'livevibe' ? '/event' : '/speak'
}

// /model as if typed. The engine queues it until the session is idle, so a switch spoken mid-turn lands when the turn ends.
// Not awaited: the sidecar loop must keep reading while the run waits for the turn.
function switchModel($: EngineInterface, model: string, isBusy: boolean) {
  if (isBusy) $.ui.toast(`live: switching to ${model} when this turn ends`)
  void $.command.run({ command: 'model', args: model })
    .then(async ({ text }) => post($, replyPath(await read($, live)), text?.trim() || `Model set to ${model}.`))
    .catch(err => $.ui.toast(`live: model switch failed: ${String(err).slice(0, 120)}`))
}

// A sidecar that is gone (it quit, or never reached ready) is not an error worth more than a debug line.
async function post($: EngineInterface, path: string, body = '') {
  const { port } = await read($, live)
  if (!port) return
  try {
    await $.http.fetch(`http://127.0.0.1:${port}${path}`, { method: 'POST', body })
  } catch (err) {
    $.ui.log(`live: POST ${path} failed: ${String(err).slice(0, 120)}`, { to: 'debug' })
  }
}

// The engine leads the line with "live-vibe: ", so it names no mode of its own: "listening · director" is live vibe,
// "voice listening" is /live, "director" alone is /vibe.
async function status($: EngineInterface) {
  const [l, v] = [await read($, live), await read($, isVibe)]
  const state = l.state.replace('_', ' ')
  const parts = [l.isOn && (l.mode === 'livevibe' ? state : `voice ${state}`), v && 'director']
  $.ui.status(parts.some(Boolean) ? parts.filter(Boolean).join(' · ') : undefined)
}

// Ends either mode: the sidecar quits, and live vibe hands vibe mode back as it found it.
async function stopLive($: EngineInterface) {
  const l = await read($, live)
  if (!l.isOn) return
  await post($, '/quit')
  if (l.mode === 'livevibe') await update($, isVibe, () => l.vibeBefore)
  await update($, live, x => ({ ...OFF, gen: x.gen || 0 }))
  await status($)
}

// Steer a running main-loop turn (the row reaches the model at its next step; nothing is aborted), or start one.
async function toClaude($: EngineInterface, text: string, steer: string) {
  const { turnId } = await read($, live)
  if (turnId) await $.session.append({ message: { type: 'user', content: [{ type: 'text', text: steer }] } })
  else void $.prompt.submit({ text, asUser: true })
}

async function onSidecar($: EngineInterface, msg: Record<string, unknown>) {
  switch (msg.type) {
    case 'ready':
      await update($, live, l => ({ ...l, port: Number(msg.port) }))
      $.ui.toast(`${(await read($, live)).mode === 'livevibe' ? 'live vibe' : 'live'}: listening (headphones recommended)`)
      break
    case 'state':
      await update($, live, l => ({ ...l, state: msg.state as LiveState }))
      await status($)
      break
    case 'utterance': {
      const text = String(msg.text)
      const model = spokenModel(text)
      if (model) {
        switchModel($, model, (await read($, live)).turnId !== null)
        break
      }
      await toClaude($, text, `[The user spoke over you: "${text}". Take this into account now: change course if it asks you to, answer it in your next reply, and keep that reply short.]`)
      break
    }
    case 'delegate': {
      const text = String(msg.text)
      await toClaude($, text, `[The user, by voice, adds a request while you work: "${text}". Take it into account now.]`)
      break
    }
    case 'switch_model':
      switchModel($, String(msg.model), (await read($, live)).turnId !== null)
      break
    case 'transcript':
      $.ui.log(`${msg.role === 'user' ? 'you (voice)' : 'voice'}: ${String(msg.text)}`)
      break
    case 'spoken':
      if (msg.cut) {
        await $.session.append({ message: { type: 'user', content: [{
          type: 'text',
          text: `[The user interrupted. Of your last answer they heard only: "${String(msg.text)}". Do not repeat it.]`,
        }] } })
      }
      break
    case 'warn':
      $.ui.toast(`live: ${String(msg.text)}`, { timeoutMs: 8000 })
      $.ui.log(`live: ${String(msg.text)}`, { to: 'debug' })
      break
    case 'log':
      $.ui.log(String(msg.text), { to: 'debug' })
      break
  }
}

// The sidecar lives for the session: the loop runs on after the hook returns and ends with the child or the module.
function startSidecar($: EngineInterface, options: PluginOptions, mode: LiveMode, vibeBefore: boolean) {
  void (async () => {
    let gen = 0
    await update($, live, (l): Live => {
      gen = (l.gen || 0) + 1
      return { isOn: true, mode, state: 'loading', port: 0, turnId: null, vibeBefore, gen }
    })
    await status($)
    const argv = ['uv', 'run', '--script', `${$.plugin.root}/bin/live_sidecar.py`,
      '--mode', mode === 'livevibe' ? 'front' : 'live', '--stt', String(options.stt), '--asr', String(options.asr),
      '--tts', String(options.tts), '--end-silence-ms', String(options.endSilenceMs)]
    if (options.voice) argv.push('--voice', String(options.voice))
    if (mode === 'livevibe') {
      argv.push('--front-backend', String(options.frontBackend), '--front-url', String(options.frontUrl),
        '--switch-pattern', SPOKEN_SWITCH.source)
      if (options.frontModel) argv.push('--front-model', String(options.frontModel))
    }
    const child = $.process.spawn({ argv })
    const isCurrent = async () => { const l = await read($, live); return l.isOn && l.gen === gen }
    let buf = ''
    for await (const chunk of child) {
      if (!('stream' in chunk)) break
      // Turned off or replaced while the child was still writing: leaving the loop kills it.
      if (!(await isCurrent())) return
      if (chunk.stream === 'stderr') { $.ui.log(chunk.text, { to: 'debug' }); continue }
      buf += chunk.text
      const lines = buf.split('\n')
      buf = lines.pop() ?? ''
      for (const line of lines) {
        if (!line.trim()) continue
        let msg: Record<string, unknown>
        try { msg = JSON.parse(line) } catch { $.ui.log(line, { to: 'debug' }); continue }
        await onSidecar($, msg)
      }
    }
    // The child ended on its own (goodbye, a crash): turn the mode off as /live or /livevibe would.
    if (await isCurrent()) {
      await update($, live, l => ({ ...l, port: 0 }))
      await stopLive($)
    }
  })().catch(err => $.ui.toast(`live: sidecar failed: ${String(err).slice(0, 120)}`))
}

// Live vibe on: vibe mode joins the voice front, and stopLive later hands vibe back as `vibeBefore` had it.
async function startLiveVibe($: EngineInterface, options: PluginOptions, vibeBefore: boolean) {
  await update($, isVibe, () => true)
  startSidecar($, options, 'livevibe', vibeBefore)
}

// What the front server offers, in a line; a server that is down or slow says so instead of holding the command.
async function servedModels($: EngineInterface, url: string) {
  try {
    const r = await Promise.race([$.http.fetch(`${url}/v1/models`), $.clock.sleep(3000).then(() => null)])
    if (!r) return `${url} did not answer within 3 s.`
    if (!r.ok) return `${url}/v1/models answered ${r.status}.`
    const ids = ((JSON.parse(r.text) as { data?: { id?: unknown }[] }).data ?? []).map(m => String(m.id))
    return `It lists: ${ids.join(', ') || 'nothing'}.`
  } catch (err) {
    return `${url} did not answer (${String(err).slice(0, 80)}).`
  }
}

// /livevibe model|url [value]: list what the front server offers, or point the front elsewhere. The model name is
// sent as each request's `model`, so a router (llama-swap, llama-server's multi-model mode) loads it on demand and a
// plain llama-server ignores it. The value is stored as this plugin's own userConfig field (`$.config.set` takes
// `<plugin>.<field>`), so it survives the session.
async function frontSetting($: EngineInterface, options: PluginOptions, field: 'model' | 'url', value: string) {
  const url = String(options.frontUrl).replace(/\/+$/, '')
  if (!value) {
    const current = `Front: ${String(options.frontBackend)} at ${url}, model ${String(options.frontModel) || '(server default)'}.`
    if (field === 'url') return { text: `${current} Change it with /livevibe url <url>.` }
    return { text: `${current} ${await servedModels($, url)} Pick one with /livevibe model <name>; /livevibe model default clears it.` }
  }
  if (field === 'url' && !/^https?:\/\/\S+$/.test(value)) return { text: `Not a URL: ${value}` }
  const next = field === 'model' && value === 'default' ? '' : value
  const { deny } = await $.config.set({ key: field === 'url' ? 'live-vibe.frontUrl' : 'live-vibe.frontModel', value: next })
  if (deny) return { text: `Front ${field} not changed: ${deny}` }
  const l = await read($, live)
  if (l.isOn && l.mode === 'livevibe') {
    // ponytail: a restart reloads Whisper and Kokoro too; a sidecar endpoint that hot-swaps the front brain is the upgrade.
    await stopLive($)
    await startLiveVibe($, { ...options, [field === 'url' ? 'frontUrl' : 'frontModel']: next }, l.vibeBefore)
    return { text: `Front ${field} set to ${next || '(server default)'}; the voice restarts on it.` }
  }
  return { text: `Front ${field} set to ${next || '(server default)'}.` }
}

export const register: Register = (on, options) => {
  on('session.start', async ($, e, next) => {
    await $.command.register({ name: 'live', description: 'Toggle live voice mode: speak to Claude, hear the answers' })
    await $.command.register({ name: 'livevibe', description: 'Toggle live vibe: talk with a fast voice front that hands the work to Claude as vibe director', argumentHint: '[model [name] | url [url]]' })
    await $.command.register({ name: 'vibe', description: 'Toggle vibe mode: Claude directs worker subagents instead of editing itself', argumentHint: '[first request]' })
    const l = await read($, live)
    if (l.isOn) startSidecar($, options, l.mode === 'livevibe' ? 'livevibe' : 'live', Boolean(l.vibeBefore)) // a hot reload killed the child; bring it back
    await status($)
    return next(e)
  })

  on('command.run', { command: 'live' }, async $ => {
    const l = await read($, live)
    await stopLive($) // off, or switching over from live vibe
    if (l.isOn && l.mode === 'live') return { text: 'Live voice off.' }
    startSidecar($, options, 'live', false)
    return { text: 'Live voice on: loading the models, then listening. Say what you want; /live again turns it off.' }
  })

  on('command.run', { command: 'livevibe' }, async ($, e) => {
    const sub = /^(model|url)(?:\s+(\S+))?$/.exec(e.args.trim())
    if (sub) return frontSetting($, options, sub[1] === 'url' ? 'url' : 'model', sub[2] ?? '')
    if (e.args.trim()) return { text: 'Usage: /livevibe toggles; /livevibe model [name]; /livevibe url [url].' }
    const l = await read($, live)
    await stopLive($) // off, or switching over from /live
    if (l.isOn && l.mode === 'livevibe') return { text: 'Live vibe off: voice front stopped, vibe mode back as it was.' }
    await startLiveVibe($, options, await read($, isVibe))
    return { text: `Live vibe on: Claude directs, a voice front (${String(options.frontBackend)}) talks with you. Loading, then listening; /livevibe again turns it off.` }
  })

  on('command.run', { command: 'vibe' }, async ($, e) => {
    const turnOn = !(await read($, isVibe))
    await update($, isVibe, () => turnOn)
    await status($)
    if (turnOn && e.args.trim()) void $.prompt.submit({ text: e.args.trim(), asUser: true })
    return { text: turnOn ? 'Vibe mode on: Claude directs, workers do the work. /vibe again exits.' : 'Vibe mode off: the full toolset is back.' }
  })

  on('prompt.compose', async ($, e, next) => {
    const { sections } = await next(e)
    const out = [...sections]
    const l = await read($, live)
    // Only the main loop lists Agent; a worker's own prompt must not be told it is the director.
    const isMain = e.tools.includes('Agent')
    if ((await read($, isVibe)) && isMain) out.push({ id: `${PLUGIN}:vibe`, text: VIBE_SECTION, scope: 'session' })
    if (l.isOn && l.mode === 'livevibe' && isMain) out.push({ id: `${PLUGIN}:relay`, text: RELAY_SECTION, scope: 'session' })
    if (l.isOn && l.mode !== 'livevibe') out.push({ id: `${PLUGIN}:voice`, text: VOICE_SECTION, scope: 'session' })
    return { sections: out }
  })

  on('tool.call', async ($, e, next) => {
    if (e.agentId || !(await read($, isVibe)) || DIRECTOR_TOOLS.has(String(e.tool))) return next(e)
    return { deny: `Vibe mode: the director does not run ${String(e.tool)}. Spawn or SendMessage a worker for it, then Read to verify.` }
  })

  // Main loop only: a subagent's run raises no turn.start.
  on('turn.start', async ($, e, next) => {
    if ((await read($, live)).isOn) await update($, live, l => ({ ...l, turnId: e.turnId }))
    return next(e)
  })

  on('turn.complete', async ($, e, next) => {
    const l = await read($, live)
    if (!l.isOn || e.agentId) return next(e)
    await update($, live, x => ({ ...x, turnId: null }))
    if (e.reason === 'answer' && e.answer.trim()) void post($, replyPath(l), e.answer)
    // The front is waiting on a result; tell it the work stopped rather than leave it silent.
    else if (l.mode === 'livevibe' && (e.reason === 'error' || e.reason === 'refusal')) {
      void post($, '/event', e.reason === 'error' ? 'The work stopped on an error before it finished.' : 'That request was declined.')
    }
    return next(e)
  })

  on('prompt.submit', async ($, e, next) => {
    // A request the user typed cuts the voice short, like a hand raised. A spoken one already did, in the sidecar,
    // and a plugin's own (a delegation, a task notification) must not silence the front mid-sentence.
    const byUser = e.origin.kind === 'composer' || e.origin.kind === 'bridge'
    if (byUser && (await read($, live)).state === 'speaking') void post($, '/stop')
    return next(e)
  })
}
