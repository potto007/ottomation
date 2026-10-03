import { atom, read, update } from 'claude-code'
import type { EngineInterface, PluginOptions, Register } from 'claude-code'

import type { Live, LiveMode, LiveState } from '../types'

const PLUGIN = 'live-vibe'
const OFF: Live = { isOn: false, mode: 'live', state: 'off', port: 0, token: '', turnId: null, vibeBefore: false, gen: 0 }
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
Live vibe is ON. The user is talking to a voice front, a small fast model that hands their spoken requests to you and relays your final answer of each turn back to them, summarized in one or two spoken sentences. End every turn with a concise plain-text report the front can summarize: what you did, what you found or changed, whether it is verified, and whether work is still running in the background (workers spawned, results pending). No spoken style needed, but lead with the outcome and leave out long listings unless asked. A bracketed note such as "[The user, by voice, adds ...]" is a live addition to act on at once. A request that starts "User said:" quotes the user's own words, then the front's reading of them; where they differ, go by the user's words. A note "[The user, by voice, to the voice front (no task asked): ...]" is information, such as a confirmation, a correction or a decision: take it as fact from the user in your next answer; it needs no reply of its own.
</live-vibe-relay>`

// A whole utterance that only asks for another model ("switch to sonnet", "change the model to opus please", "use haiku").
// Anchored at both ends, so a request that merely mentions a model ("use opus to review this") stays a prompt.
// The live vibe sidecar receives this source as --switch-pattern, so both modes match the same words.
const SPOKEN_SWITCH = /^(?:(?:ok|okay|hey|alright) )?(?:claude )?(?:please )?(?:(?:switch|change|swap|set)(?: over)?(?: the)?(?: model)? to|use)(?: the)? (opus|sonnet|haiku|fable)(?: model)?(?: please)?$/

function spokenModel(text: string): string | undefined {
  return SPOKEN_SWITCH.exec(text.toLowerCase().replace(/[^a-z ]+/g, ' ').replace(/\s+/g, ' ').trim())?.[1]
}

// Claude's whole answer to a voice question when the front's own holding line already says enough: never posted.
const NOTHING_TO_ADD = '(nothing to add)'

function nothingToAdd(answer: string) {
  return answer.trim().toLowerCase().replace(/\./g, '') === NOTHING_TO_ADD
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
// The sidecar refuses a POST without the token it reported in `ready`, so no other local process can drive it.
async function post($: EngineInterface, path: string, body = '') {
  const { port, token } = await read($, live)
  if (!port) return
  try {
    await $.http.fetch(`http://127.0.0.1:${port}${path}`, { method: 'POST', body, headers: { 'X-Live-Token': token } })
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
      await update($, live, l => ({ ...l, port: Number(msg.port), token: String(msg.token ?? '') }))
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
    // The front's reading of a request, with the user's own words when the sidecar sends them (`said`): a small
    // model's rewrite can drop the question, so Claude always sees what was actually said.
    case 'delegate': {
      const text = String(msg.text)
      const said = typeof msg.said === 'string' ? msg.said.trim() : ''
      if (!said) {
        await toClaude($, text, `[The user, by voice, adds a request while you work: "${text}". Take it into account now.]`)
        break
      }
      await toClaude($, `User said: "${said}"\nThe voice front read it as: ${text}`,
        `[The user, by voice, adds while you work: "${said}". The voice front read it as: ${text}. Take it into account now.]`)
      break
    }
    // A user turn the front answered itself: a confirmation, a correction or a decision is still Claude's to know.
    // It joins the conversation without starting a turn. Two kinds are prompts: a reply to the question Claude's last
    // answer asked (`answer`), and a question the front did not delegate (`ask`): it said only a holding line, and
    // Claude gives the real answer, or NOTHING_TO_ADD, which turn.complete keeps from the front.
    case 'note': {
      const said = typeof msg.said === 'string' ? msg.said.trim() : ''
      if (!said) break
      const reply = typeof msg.reply === 'string' && msg.reply.trim() ? ` (the voice front answered: "${msg.reply.trim()}")` : ''
      if (msg.answer === true) {
        await toClaude($, `User said, answering your last question: "${said}"`,
          `[The user, by voice, answers your last question: "${said}"${reply}. Take it into account now.]`)
        break
      }
      if (msg.ask === true) {
        await toClaude($, `User asked by voice: "${said}"${reply}\nThe voice front did not answer it. Give the real answer. If you have nothing to add beyond the voice front's line, reply with exactly: ${NOTHING_TO_ADD}`,
          `[The user, by voice, asks while you work: "${said}"${reply}. The voice front did not answer it: answer it in your next reply.]`)
        break
      }
      await $.session.append({ message: { type: 'user', content: [{
        type: 'text', text: `[The user, by voice, to the voice front (no task asked): "${said}"${reply}]`,
      }] } })
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

// With frontUrl empty the sidecar runs its own llama-server; these settings point it at files already on disk.
function frontServerArgv(options: PluginOptions) {
  const argv: string[] = []
  if (options.frontServerBin) argv.push('--front-server-bin', String(options.frontServerBin))
  if (options.frontServerModel) argv.push('--front-server-model', String(options.frontServerModel))
  if (options.frontServerLog) argv.push('--front-server-log', String(options.frontServerLog))
  return argv
}

// The sidecar lives for the session: the loop runs on after the hook returns and ends with the child or the module.
function startSidecar($: EngineInterface, options: PluginOptions, mode: LiveMode, vibeBefore: boolean) {
  void (async () => {
    let gen = 0
    await update($, live, (l): Live => {
      gen = (l.gen || 0) + 1
      return { isOn: true, mode, state: 'loading', port: 0, token: '', turnId: null, vibeBefore, gen }
    })
    await status($)
    const argv = ['uv', 'run', '--script', `${$.plugin.root}/bin/sidecar/main.py`,
      '--mode', mode === 'livevibe' ? 'front' : 'live', '--stt', String(options.stt), '--asr', String(options.asr),
      '--tts', String(options.tts), '--end-silence-ms', String(options.endSilenceMs)]
    if (options.voice) argv.push('--voice', String(options.voice))
    if (options.mic) argv.push('--mic', String(options.mic))
    if (options.speaker) argv.push('--speaker', String(options.speaker))
    if (options.speakerBackend) argv.push('--speaker-backend', String(options.speakerBackend))
    if (options.logFile) argv.push('--log-file', String(options.logFile))
    if (mode === 'livevibe') {
      argv.push('--front-backend', String(options.frontBackend), '--front-url', String(options.frontUrl),
        '--switch-pattern', SPOKEN_SWITCH.source, ...frontServerArgv(options))
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
  })().catch(err => sidecarFailed($, err).catch(() => {})) // a module unloading mid-loop lands here too: stay quiet
}

// The child never started (no uv, most often) or the loop broke: say why, and turn off a mode that never got going.
async function sidecarFailed($: EngineInterface, err: unknown) {
  $.ui.toast(await hasUv($) ? `live: sidecar failed: ${String(err).slice(0, 120)}. /live setup checks everything.` : `live: ${UV_FIX}`, { timeoutMs: 15000 })
  const l = await read($, live)
  if (l.isOn && !l.port) await stopLive($)
}

const UV_FIX = 'uv is not installed, or not on the PATH Claude Code started with. Install it (curl -LsSf https://astral.sh/uv/install.sh | sh; macOS also brew install uv; Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"), restart Claude Code, then /live setup.'

async function hasUv($: EngineInterface) {
  try {
    return (await $.process.run(['uv', '--version'])).exitCode === 0
  } catch {
    return false
  }
}

const MARK: Record<string, string> = { ok: '✓', warn: '!', fail: '✗' }
let isSettingUp = false // one at a time; a reload kills the child and starts this over

// /live setup: the sidecar's --setup installs the Python packages (uv does, before Python starts), downloads the
// models the settings name, and tests the mic, speaker, synthesizer, recognizer and front. Progress goes to the
// status line; the report is one transcript row when it ends. System packages stay the person's to install: the
// report names the command.
function setupVoice($: EngineInterface, options: PluginOptions) {
  isSettingUp = true
  void (async () => {
    $.ui.status('setup: installing the Python packages (a minute or two on first run)')
    const argv = ['uv', 'run', '--script', `${$.plugin.root}/bin/sidecar/main.py`, '--setup',
      '--stt', String(options.stt), '--asr', String(options.asr), '--tts', String(options.tts),
      '--end-silence-ms', String(options.endSilenceMs), '--front-backend', String(options.frontBackend),
      '--front-url', String(options.frontUrl), ...frontServerArgv(options)]
    if (options.voice) argv.push('--voice', String(options.voice))
    if (options.mic) argv.push('--mic', String(options.mic))
    if (options.speaker) argv.push('--speaker', String(options.speaker))
    if (options.speakerBackend) argv.push('--speaker-backend', String(options.speakerBackend))
    if (options.logFile) argv.push('--log-file', String(options.logFile))
    if (options.frontModel) argv.push('--front-model', String(options.frontModel))
    const rows: string[] = []
    const errTail: string[] = []
    let ok: boolean | null = null
    let buf = ''
    for await (const chunk of $.process.spawn({ argv })) {
      if (!('stream' in chunk)) break
      if (chunk.stream === 'stderr') {
        $.ui.log(chunk.text, { to: 'debug' })
        errTail.push(...chunk.text.split('\n').filter(x => x.trim()))
        errTail.splice(0, Math.max(0, errTail.length - 8))
        continue
      }
      buf += chunk.text
      const lines = buf.split('\n')
      buf = lines.pop() ?? ''
      for (const line of lines) {
        let msg: Record<string, unknown>
        try { msg = JSON.parse(line) } catch { continue }
        const text = String(msg.text ?? '')
        if (msg.type === 'progress') $.ui.status(`setup: ${text}`)
        else if (msg.type === 'check') rows.push(`${MARK[String(msg.status)] ?? '?'} ${String(msg.name)}: ${text}`)
        else if (msg.type === 'warn') rows.push(`! ${text}`)
        else if (msg.type === 'log' && /^(download|kyutai):/.test(text)) $.ui.status(`setup: ${text}`)
        else if (msg.type === 'done') ok = Boolean(msg.ok)
      }
    }
    const head = ok ? 'Live voice setup: ready. /live or /livevibe to start.'
      : ok === false ? 'Live voice setup: fix the ✗ lines, then /live setup again.'
        : 'Live voice setup stopped before it finished. The last output:'
    $.ui.log([head, ...rows, ...(ok === null ? errTail : [])].join('\n'))
    $.ui.toast(ok ? 'live setup: ready' : 'live setup: needs attention (see the transcript)')
  })()
    .catch(err => $.ui.toast(`live setup failed: ${String(err).slice(0, 160)}`))
    .finally(() => { isSettingUp = false; void status($).catch(() => {}) })
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
// `<plugin>.<field>`), so it survives the session. `/livevibe url managed` empties frontUrl: the sidecar's own server.
async function frontSetting($: EngineInterface, options: PluginOptions, field: 'model' | 'url', value: string) {
  const url = String(options.frontUrl ?? '').replace(/\/+$/, '')
  if (!value) {
    const at = url ? `at ${url}` : `on a managed llama-server (${String(options.frontServerModel || 'the default model')})`
    const current = `Front: ${String(options.frontBackend)} ${at}, model ${String(options.frontModel) || '(server default)'}.`
    if (field === 'url') return { text: `${current} Change it with /livevibe url <url>, or /livevibe url managed.` }
    if (!url) return { text: `${current} The managed server serves one model; frontServerModel picks it.` }
    return { text: `${current} ${await servedModels($, url)} Pick one with /livevibe model <name>; /livevibe model default clears it.` }
  }
  if (field === 'url' && value !== 'managed' && !/^https?:\/\/\S+$/.test(value)) return { text: `Not a URL: ${value}` }
  const next = (field === 'model' && value === 'default') || (field === 'url' && value === 'managed') ? '' : value
  const { deny } = await $.config.set({ key: field === 'url' ? 'live-vibe.frontUrl' : 'live-vibe.frontModel', value: next })
  if (deny) return { text: `Front ${field} not changed: ${deny}` }
  const l = await read($, live)
  if (l.isOn && l.mode === 'livevibe') {
    // ponytail: a restart reloads Whisper and Kokoro too; a sidecar endpoint that hot-swaps the front brain is the upgrade.
    await stopLive($)
    await startLiveVibe($, { ...options, [field === 'url' ? 'frontUrl' : 'frontModel']: next }, l.vibeBefore)
    return { text: `Front ${field} set to ${next || (field === 'url' ? '(managed llama-server)' : '(server default)')}; the voice restarts on it.` }
  }
  return { text: `Front ${field} set to ${next || (field === 'url' ? '(managed llama-server)' : '(server default)')}.` }
}

export const register: Register = (on, options) => {
  on('session.start', async ($, e, next) => {
    await $.command.register({ name: 'live', description: 'Toggle live voice mode: speak to Claude, hear the answers. /live setup installs and tests what it needs', argumentHint: '[setup]' })
    await $.command.register({ name: 'livevibe', description: 'Toggle live vibe: talk with a fast voice front that hands the work to Claude as vibe director', argumentHint: '[model [name] | url [url|managed]]' })
    await $.command.register({ name: 'vibe', description: 'Toggle vibe mode: Claude directs worker subagents instead of editing itself', argumentHint: '[first request]' })
    const l = await read($, live)
    if (l.isOn) startSidecar($, options, l.mode === 'livevibe' ? 'livevibe' : 'live', Boolean(l.vibeBefore)) // a hot reload killed the child; bring it back
    await status($)
    return next(e)
  })

  on('command.run', { command: 'live' }, async ($, e) => {
    const args = e.args.trim()
    if (args && args !== 'setup') return { text: 'Usage: /live toggles live voice; /live setup checks and installs what it needs.' }
    if (args === 'setup') {
      if ((await read($, live)).isOn) return { text: 'Turn /live or /livevibe off first: setup loads the same models.' }
      if (isSettingUp) return { text: 'Setup is already running; progress is in the status line.' }
      if (!(await hasUv($))) return { text: UV_FIX }
      setupVoice($, options)
      return { text: 'Setting up live voice: Python packages, models, then a test of the mic, speaker and voice. Progress is in the status line; the report lands here.' }
    }
    const l = await read($, live)
    await stopLive($) // off, or switching over from live vibe
    if (l.isOn && l.mode === 'live') return { text: 'Live voice off.' }
    startSidecar($, options, 'live', false)
    return { text: 'Live voice on: loading the models, then listening. Say what you want; /live again turns it off. First run on this machine? /live setup shows progress.' }
  })

  on('command.run', { command: 'livevibe' }, async ($, e) => {
    const sub = /^(model|url)(?:\s+(\S+))?$/.exec(e.args.trim())
    if (sub) return frontSetting($, options, sub[1] === 'url' ? 'url' : 'model', sub[2] ?? '')
    if (e.args.trim()) return { text: 'Usage: /livevibe toggles; /livevibe model [name]; /livevibe url [url|managed].' }
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
    if (e.reason === 'answer' && e.answer.trim()) {
      if (!nothingToAdd(e.answer)) void post($, replyPath(l), e.answer)
    }
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
