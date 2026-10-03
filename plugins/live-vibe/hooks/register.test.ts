import { expect, test } from 'claude-code/testing'

const typed = { args: '', origin: { kind: 'composer' }, presentation: { isFullscreen: false, columns: 120 } } as const
const compose = (tools: string[]) =>
  ({ model: 'm', promptModel: 'm', surfaces: ['terminal'], tools, outputStyle: null, traits: [] }) as const

test('/vibe toggles, denies the director Bash, lets a worker and Read through', async ($, on) => {
  on('tool.call', () => ({ result: 'ran' }))
  on('prompt.compose', () => ({ sections: [] }))

  expect((await $.command.run({ command: 'vibe', ...typed })).text).toMatch(/on/)

  const denied = await $.tool.call({ tool: 'Bash', command: 'true' })
  expect('deny' in denied && denied.deny).toMatch(/Vibe mode/)
  expect('deny' in (await $.tool.call({ tool: 'Read', file_path: '/dev/null' }))).toBe(false)

  const main = await $.prompt.compose(compose(['Read', 'Agent', 'Bash']))
  expect(main.sections.some(s => s.id === 'live-vibe:vibe')).toBe(true)
  const worker = await $.prompt.compose(compose(['Read', 'Bash']))
  expect(worker.sections.some(s => s.id === 'live-vibe:vibe')).toBe(false)

  expect((await $.command.run({ command: 'vibe', ...typed })).text).toMatch(/off/)
  expect('deny' in (await $.tool.call({ tool: 'Bash', command: 'true' }))).toBe(false)
  expect((await $.prompt.compose(compose(['Agent']))).sections.length).toBe(0)
})

test('a spoken model switch runs /model; other speech is still a prompt', async ($, on) => {
  on('process.spawn', async function* () {
    yield { stream: 'stdout' as const, text: '{"type":"utterance","text":"Okay, switch to Sonnet."}\n' }
    yield { stream: 'stdout' as const, text: '{"type":"utterance","text":"Use opus to review this."}\n' }
    return { value: { code: 0, signal: null } }
  })
  const switched = new Promise<string>(resolve =>
    on('command.run', { command: 'model' }, (_$, e) => { resolve(e.args); return { text: 'Set model to Sonnet' } }))
  const prompted = new Promise<string>(resolve =>
    on('prompt.submit', (_$, e) => { resolve(e.text); return { text: e.text } }))

  await $.command.run({ command: 'live', ...typed })

  expect(await switched).toBe('sonnet')
  expect(await prompted).toBe('Use opus to review this.')
})

// A sidecar that reports ready on `port`, then the given lines, then stays up until `release()`.
function fakeSidecar(port: number, ...lines: string[]) {
  let release = () => {}
  let up = () => {}
  const isUp = new Promise<void>(resolve => { up = resolve })
  const held = new Promise<void>(resolve => { release = resolve })
  const spawn = async function* () {
    yield { stream: 'stdout' as const, text: `{"type":"ready","port":${port},"token":"tok-${port}"}\n` }
    for (const line of lines) yield { stream: 'stdout' as const, text: `${line}\n` }
    yield { stream: 'stdout' as const, text: '{"type":"state","state":"listening"}\n' }
    up() // every line above has been handled once the loop pulls past them
    await held
    return { value: { code: 0, signal: null } }
  }
  return { spawn, isUp, release }
}

const ok = { value: { status: 204, ok: true, headers: {}, text: '' } }

test('/livevibe turns on vibe and the relay prompt, and off restores both; /live switches over', async ($, on) => {
  const posts: string[] = []
  const lines: (string | undefined)[] = []
  const sidecar = fakeSidecar(4321)
  on('process.spawn', sidecar.spawn)
  on('http.fetch', (_$, e) => { posts.push(e.url); return ok })
  on('ui.status', (_$, e) => { lines.push(e.text); return { value: undefined } })
  on('tool.call', () => ({ result: 'ran' }))
  on('prompt.compose', () => ({ sections: [] }))
  const ids = async () => (await $.prompt.compose(compose(['Read', 'Agent', 'Bash']))).sections.map(s => s.id)

  expect((await $.command.run({ command: 'livevibe', ...typed })).text).toMatch(/Live vibe on/)
  await sidecar.isUp
  expect('deny' in (await $.tool.call({ tool: 'Bash', command: 'true' }))).toBe(true)
  expect(await ids()).toEqual(['live-vibe:vibe', 'live-vibe:relay'])
  expect(lines.at(-1)).toBe('listening · director')

  expect((await $.command.run({ command: 'livevibe', ...typed })).text).toMatch(/Live vibe off/)
  expect(posts).toEqual(['http://127.0.0.1:4321/quit'])
  expect('deny' in (await $.tool.call({ tool: 'Bash', command: 'true' }))).toBe(false)
  expect(await ids()).toEqual([])
  expect(lines.at(-1)).toBe(undefined)

  // On again, then /live takes over: the front quits, vibe goes back off, the voice prompt replaces the relay.
  await $.command.run({ command: 'livevibe', ...typed })
  await $.command.run({ command: 'live', ...typed })
  expect(await ids()).toEqual(['live-vibe:voice'])
  expect('deny' in (await $.tool.call({ tool: 'Bash', command: 'true' }))).toBe(false)
  await $.command.run({ command: 'live', ...typed })
  sidecar.release()
})

test('live vibe: a delegate from the front becomes a prompt when Claude is idle', async ($, on) => {
  const sidecar = fakeSidecar(4321, '{"type":"delegate","text":"Fix the failing test in parser.py"}')
  on('process.spawn', sidecar.spawn)
  on('http.fetch', () => ok)
  const prompted = new Promise<{ text: string; origin: string }>(resolve =>
    on('prompt.submit', (_$, e) => { resolve({ text: e.text, origin: e.origin.kind }); return { text: e.text } }))

  await $.command.run({ command: 'livevibe', ...typed })

  expect(await prompted).toEqual({ text: 'Fix the failing test in parser.py', origin: 'plugin' })
  await $.command.run({ command: 'livevibe', ...typed })
  sidecar.release()
})

test("live vibe: a delegate carries the user's own words beside the front's reading", async ($, on) => {
  const sidecar = fakeSidecar(4321, '{"type":"delegate","text":"Summarize the log report","said":"So what was the fix?"}')
  on('process.spawn', sidecar.spawn)
  on('http.fetch', () => ok)
  const prompted = new Promise<string>(resolve =>
    on('prompt.submit', (_$, e) => { resolve(e.text); return { text: e.text } }))

  await $.command.run({ command: 'livevibe', ...typed })

  expect(await prompted).toBe('User said: "So what was the fix?"\nThe voice front read it as a task: Summarize the log '
    + "report.\nAnswer the user's words; treat the front's reading as a hint, and if it is a question about status or "
    + 'progress, answer from what you know rather than starting new work.')
  await $.command.run({ command: 'livevibe', ...typed })
  sidecar.release()
})

// The live prompt of 2026-10-03: a reading that ends in a full stop got another one ("methods.. Answer").
test('live vibe: a delegation prompt has one full stop after the reading and the instructions once', async ($, on) => {
  const sidecar = fakeSidecar(4321, JSON.stringify({ type: 'delegate', said: 'Can you have an agent research this?',
    text: 'Research end-of-utterance detection and real-time methods.' }))
  on('process.spawn', sidecar.spawn)
  on('http.fetch', () => ok)
  const prompted = new Promise<string>(resolve =>
    on('prompt.submit', (_$, e) => { resolve(e.text); return { text: e.text } }))

  await $.command.run({ command: 'livevibe', ...typed })
  const text = await prompted

  expect(text).toContain('read it as a task: Research end-of-utterance detection and real-time methods.\nAnswer')
  expect(text).not.toContain('..')
  expect(text.split('rather than starting new work').length).toBe(2)
  await $.command.run({ command: 'livevibe', ...typed })
  sidecar.release()
})

test('live vibe: a note joins the conversation without a turn; an answer or a question is a prompt', async ($, on) => {
  const sidecar = fakeSidecar(4321,
    '{"type":"note","said":"It sounds perfect. It is fixed.","reply":"Great, the static is gone."}',
    '{"type":"note","said":"Yes, delete it.","reply":"Okay.","answer":true}',
    '{"type":"note","said":"So what was the fix?","reply":"Let me check.","ask":true}')
  on('process.spawn', sidecar.spawn)
  on('http.fetch', () => ok)
  const appended: string[] = []
  on('session.append', (_$, e) => {
    const block = e.message.content[0]
    if (e.door === 'note' && block && 'text' in block) appended.push(String(block.text))
    return { message: e.message, uuid: e.uuid }
  })
  const prompted: string[] = []
  on('prompt.submit', (_$, e) => { prompted.push(e.text); return { text: e.text } })

  await $.command.run({ command: 'livevibe', ...typed })
  await sidecar.isUp

  expect(appended).toEqual(['[The user, by voice, to the voice front (no task asked): "It sounds perfect. It is fixed." '
    + '(the voice front answered: "Great, the static is gone.")]'])
  expect(prompted).toEqual(['User said, answering your last question: "Yes, delete it."',
    'User asked by voice: "So what was the fix?" (the voice front answered: "Let me check.")\n'
    + "The voice front did not answer it. Give the real answer. If you have nothing to add beyond the voice front's "
    + 'line, reply with exactly: (nothing to add)'])
  await $.command.run({ command: 'livevibe', ...typed })
  sidecar.release()
})

test("live vibe: Claude's answer goes to the front's /event, never /speak", async ($, on) => {
  const posts: { url: string; body?: string; token?: string }[] = []
  const sidecar = fakeSidecar(4321)
  on('process.spawn', sidecar.spawn)
  on('turn.complete', (_$, e) => ({ text: e.answer }))
  const relayed = new Promise<void>(resolve => on('http.fetch', (_$, e) => {
    posts.push({ url: e.url, body: e.init?.body, token: e.init?.headers?.['X-Live-Token'] })
    if (e.url.endsWith('/event')) resolve()
    return ok
  }))

  await $.command.run({ command: 'livevibe', ...typed })
  await sidecar.isUp
  await $.turn.complete({ answer: 'Fixed parser.py; all 41 tests pass.', durationMs: 1, isAborted: false, turnId: 't1', reason: 'answer' })
  await relayed

  expect(posts).toEqual([{ url: 'http://127.0.0.1:4321/event', body: 'Fixed parser.py; all 41 tests pass.', token: 'tok-4321' }])
  await $.command.run({ command: 'livevibe', ...typed })
  sidecar.release()
})

test("live vibe: Claude's '(nothing to add)' never reaches the front", async ($, on) => {
  const posts: { url: string; body?: string }[] = []
  const sidecar = fakeSidecar(4321)
  on('process.spawn', sidecar.spawn)
  on('turn.complete', (_$, e) => ({ text: e.answer }))
  const relayed = new Promise<void>(resolve => on('http.fetch', (_$, e) => {
    posts.push({ url: e.url, body: e.init?.body })
    if (e.url.endsWith('/event')) resolve()
    return ok
  }))

  await $.command.run({ command: 'livevibe', ...typed })
  await sidecar.isUp
  await $.turn.complete({ answer: ' (nothing to add) ', durationMs: 1, isAborted: false, turnId: 't1', reason: 'answer' })
  await $.turn.complete({ answer: '(Nothing to add.)', durationMs: 1, isAborted: false, turnId: 't2', reason: 'answer' })
  await $.turn.complete({ answer: 'Still running; nothing to add yet.', durationMs: 1, isAborted: false, turnId: 't3', reason: 'answer' })
  await relayed

  expect(posts).toEqual([{ url: 'http://127.0.0.1:4321/event', body: 'Still running; nothing to add yet.' }])
  await $.command.run({ command: 'livevibe', ...typed })
  sidecar.release()
})

test('/livevibe model lists what the front server serves, and /livevibe model foo sets the field', { options: { frontUrl: 'http://127.0.0.1:8080' } }, async ($, on) => {
  const fetched: string[] = []
  on('http.fetch', (_$, e) => {
    fetched.push(e.url)
    return { value: { status: 200, ok: true, headers: {}, text: '{"object":"list","data":[{"id":"qwen3-9b"},{"id":"qwen3.6-35b-a3b"}]}' } }
  })
  on('clock.sleep', () => new Promise(() => {})) // the 3 s timeout never fires; the server answers first
  const sets: { key: string; value: unknown }[] = []
  on('config.set', (_$, e) => { sets.push({ key: e.key, value: e.value }); return { value: e.value } })

  const listed = (await $.command.run({ command: 'livevibe', ...typed, args: 'model' })).text
  expect(fetched).toEqual(['http://127.0.0.1:8080/v1/models'])
  expect(listed).toMatch(/qwen3-9b, qwen3\.6-35b-a3b/)

  expect((await $.command.run({ command: 'livevibe', ...typed, args: 'model foo' })).text).toMatch(/set to foo/)
  await $.command.run({ command: 'livevibe', ...typed, args: 'url http://gpu-box:8080' })
  await $.command.run({ command: 'livevibe', ...typed, args: 'model default' })
  expect(sets).toEqual([
    { key: 'live-vibe.frontModel', value: 'foo' },
    { key: 'live-vibe.frontUrl', value: 'http://gpu-box:8080' },
    { key: 'live-vibe.frontModel', value: '' },
  ])
})

test('/livevibe model foo while live vibe is on restarts the front on foo, vibe still on', async ($, on) => {
  const argvs: (readonly string[])[] = []
  const sidecar = fakeSidecar(4321)
  on('process.spawn', async function* (_$, e) { argvs.push(e.argv); return yield* sidecar.spawn() })
  const posts: string[] = []
  on('http.fetch', (_$, e) => { posts.push(e.url); return ok })
  on('config.set', (_$, e) => ({ value: e.value }))
  on('tool.call', () => ({ result: 'ran' }))

  await $.command.run({ command: 'livevibe', ...typed })
  await sidecar.isUp
  expect((await $.command.run({ command: 'livevibe', ...typed, args: 'model foo' })).text).toMatch(/restarts/)
  expect(posts).toEqual(['http://127.0.0.1:4321/quit'])
  expect('deny' in (await $.tool.call({ tool: 'Bash', command: 'true' }))).toBe(true)
  await $.command.run({ command: 'livevibe', ...typed }) // off: the second fake reports no port, so nothing more is posted
  expect(argvs.length).toBe(2)
  expect(argvs[0]).not.toContain('--front-model')
  expect(argvs[1]?.join(' ')).toContain('--front-model foo')
  sidecar.release()
})

test('frontUrl empty: a managed llama-server, its settings reach the sidecar, /livevibe url managed clears the URL',
  { options: { frontServerBin: '/opt/llama/llama-server', frontServerLog: '/tmp/x.log' } }, async ($, on) => {
    const fetched: string[] = []
    on('http.fetch', (_$, e) => { fetched.push(e.url); return ok })
    const sets: { key: string; value: unknown }[] = []
    on('config.set', (_$, e) => { sets.push({ key: e.key, value: e.value }); return { value: e.value } })
    const sidecar = fakeSidecar(4321)
    const argv = new Promise<string>(resolve =>
      on('process.spawn', async function* (_$, e) { resolve(e.argv.join(' ')); return yield* sidecar.spawn() }))
    on('tool.call', () => ({ result: 'ran' }))

    expect((await $.command.run({ command: 'livevibe', ...typed, args: 'model' })).text).toMatch(/managed llama-server \(the default model\)/)
    expect(fetched).toEqual([])
    expect((await $.command.run({ command: 'livevibe', ...typed, args: 'url managed' })).text).toMatch(/managed llama-server/)
    expect(sets).toEqual([{ key: 'live-vibe.frontUrl', value: '' }])
    await $.command.run({ command: 'livevibe', ...typed })
    const got = await argv
    expect(got).toContain('--front-url  --switch-pattern')
    expect(got).toContain('--front-server-bin /opt/llama/llama-server --front-server-log /tmp/x.log')
    expect(got).not.toContain('--front-server-model')
    await sidecar.isUp
    await $.command.run({ command: 'livevibe', ...typed })
    sidecar.release()
  })

test('stt defaults to kyutai, tts to kokoro', async ($, on) => {
  const argvs: string[] = []
  const sidecar = fakeSidecar(4321)
  on('process.spawn', async function* (_$, e) { argvs.push(e.argv.join(' ')); return yield* sidecar.spawn() })
  on('http.fetch', () => ok)

  await $.command.run({ command: 'live', ...typed })
  await $.command.run({ command: 'live', ...typed })
  expect(argvs[0]).toContain('--stt kyutai')
  expect(argvs[0]).toContain('--tts kokoro')
  sidecar.release()
})

test('stt: whisper reaches the sidecar argv', { options: { stt: 'whisper' } }, async ($, on) => {
  const sidecar = fakeSidecar(4321)
  const argv = new Promise<string>(resolve =>
    on('process.spawn', async function* (_$, e) { resolve(e.argv.join(' ')); return yield* sidecar.spawn() }))
  on('http.fetch', () => ok)

  await $.command.run({ command: 'live', ...typed })
  expect(await argv).toContain('--stt whisper')
  await $.command.run({ command: 'live', ...typed })
  sidecar.release()
})

test('/live setup runs the sidecar --setup and logs its report', async ($, on) => {
  on('process.run', () => ({ value: { exitCode: 0, stdout: 'uv 0.9.0\n', stderr: '', isStdoutTruncated: false, isStderrTruncated: false } }))
  const argvs: string[] = []
  on('process.spawn', async function* (_$, e) {
    argvs.push(e.argv.join(' '))
    yield { stream: 'stdout' as const, text: '{"type":"progress","text":"Kokoro: loading"}\n{"type":"check","name":"audio","status":"ok","text":"mic A"}\n' }
    yield { stream: 'stdout' as const, text: '{"type":"check","name":"espeak-ng","status":"fail","text":"brew install espeak-ng"}\n{"type":"done","ok":false}\n' }
    return { value: { code: 1, signal: null } }
  })
  const statuses: (string | undefined)[] = []
  on('ui.status', (_$, e) => { statuses.push(e.text); return { value: undefined } })
  const report = new Promise<string>(resolve => on('ui.log', (_$, e) => { if (e.to !== 'debug') resolve(e.text); return { value: undefined } }))

  expect((await $.command.run({ command: 'live', ...typed, args: 'setup' })).text).toMatch(/Setting up live voice/)
  const text = await report
  expect(argvs[0]).toContain('--setup')
  expect(text).toMatch(/fix the ✗ lines/)
  expect(text).toContain('✓ audio: mic A')
  expect(text).toContain('✗ espeak-ng: brew install espeak-ng')
  expect(statuses).toContain('setup: Kokoro: loading')
})

test('/live setup without uv says how to install it; /live with other words shows usage', async ($, on) => {
  on('process.run', () => { throw new Error('spawn uv ENOENT') })
  expect((await $.command.run({ command: 'live', ...typed, args: 'setup' })).text).toMatch(/uv is not installed/)
  expect((await $.command.run({ command: 'live', ...typed, args: 'please' })).text).toMatch(/Usage/)
})

// A sidecar that reports ready, sends `first`, waits for `go()`, sends `later`, and says when each batch is handled.
function scriptedSidecar(first: string[], later: string[]) {
  let go = () => {}
  let firstDone = () => {}
  let laterDone = () => {}
  const gate = new Promise<void>(resolve => { go = resolve })
  const sentFirst = new Promise<void>(resolve => { firstDone = resolve })
  const sentLater = new Promise<void>(resolve => { laterDone = resolve })
  let release = () => {}
  const held = new Promise<void>(resolve => { release = resolve })
  const state = '{"type":"state","state":"listening"}\n'
  const spawn = async function* () {
    yield { stream: 'stdout' as const, text: '{"type":"ready","port":4321,"token":"tok"}\n' }
    for (const line of first) yield { stream: 'stdout' as const, text: `${line}\n` }
    yield { stream: 'stdout' as const, text: state }
    firstDone()
    await gate
    for (const line of later) yield { stream: 'stdout' as const, text: `${line}\n` }
    yield { stream: 'stdout' as const, text: state }
    laterDone()
    await held
    return { value: { code: 0, signal: null } }
  }
  return { spawn, go, sentFirst, sentLater, release }
}

const said = (role: string, text: string, kind?: string) => JSON.stringify({ type: 'transcript', role, text, kind })

test('live vibe: fillers share one short line, a retelling of the answer above is one clipped line', async ($, on) => {
  const answer = 'Research is launched. A worker is surveying what production voice agents do now for end of turn, '
    + 'from the papers and the vendor docs. One worker is running.'
  const sidecar = scriptedSidecar([
    said('user', 'Can you check the logs?'),
    said('front', 'Let me check.', 'filler'),
    said('front', 'Let me check.', 'filler'),
    said('front', "On it, I've asked.", 'filler'),
    said('user', 'Thanks.'),
    said('front', "You're welcome.", 'reply'),
  ], [
    said('front', 'Research is launched. A worker is surveying what production voice agents do now for end of turn.', 'relay'),
    said('front', 'The build passed and nothing is pushed.', 'relay'),
  ])
  on('process.spawn', sidecar.spawn)
  on('http.fetch', () => ok)
  on('clock.sleep', () => new Promise(() => {})) // a filler's line waits for the next line, not the timer
  on('turn.complete', (_$, e) => ({ text: e.answer }))
  const shown: string[] = []
  const debug: string[] = []
  on('ui.log', (_$, e) => { (e.to === 'debug' ? debug : shown).push(e.text); return { value: undefined } })

  await $.command.run({ command: 'livevibe', ...typed })
  await sidecar.sentFirst
  await $.turn.complete({ answer, durationMs: 1, isAborted: false, turnId: 't1', reason: 'answer' })
  sidecar.go()
  await sidecar.sentLater

  expect(shown.slice(0, 4)).toEqual(['you (voice): Can you check the logs?', "spoken: Let me check. (x2) / On it, I've asked.",
    'you (voice): Thanks.', "voice: You're welcome."])
  expect(shown[4]).toMatch(/^spoken summary: Research is launched\. A worker .*\u2026 \(repeats the answer above\)$/)
  expect(shown[4]!.length).toBeLessThan(120)
  expect(shown[5]).toBe('spoken summary: The build passed and nothing is pushed.')
  expect(shown.length).toBe(6)
  expect(debug.some(d => d.includes('spoken in full: Research is launched. A worker is surveying'))).toBe(true)
  await $.command.run({ command: 'livevibe', ...typed })
  sidecar.release()
})

test("live vibe: a filler alone gets its line when the timer runs out; one Claude's answer beat is dropped", async ($, on) => {
  const sidecar = scriptedSidecar([said('front', 'Let me check.', 'filler')], [said('front', 'On it.', 'filler')])
  on('process.spawn', sidecar.spawn)
  on('http.fetch', () => ok)
  on('turn.complete', (_$, e) => ({ text: e.answer }))
  let wake = () => {}
  on('clock.sleep', () => new Promise(resolve => { wake = () => resolve({ value: undefined }) }))
  const shown: string[] = []
  const debug: string[] = []
  let logged = () => {}
  on('ui.log', (_$, e) => { (e.to === 'debug' ? debug : shown).push(e.text); logged(); return { value: undefined } })

  await $.command.run({ command: 'livevibe', ...typed })
  await sidecar.sentFirst
  expect(shown).toEqual([])
  const landed = new Promise<void>(resolve => { logged = resolve })
  wake()
  await landed
  expect(shown).toEqual(['spoken: Let me check.'])

  sidecar.go()
  await sidecar.sentLater
  await $.turn.complete({ answer: 'Done: all 41 pass.', durationMs: 1, isAborted: false, turnId: 't1', reason: 'answer' })
  expect(shown).toEqual(['spoken: Let me check.'])
  expect(debug.some(d => d.includes('spoken: On it. (dropped: the answer came first)'))).toBe(true)
  await $.command.run({ command: 'livevibe', ...typed })
  sidecar.release()
})

const prompt = 'User said: "Can you research this?"\nThe voice front read it as a task: Research end-of-turn detection.\n'
  + "Answer the user's words; treat the front's reading as a hint."

for (const surface of ['terminal', 'desktop'] as const) {
  test(`live vibe: its voice prompt is one dim line until expanded (${surface})`, async ($, on) => {
    on('ui.render', () => ({ type: 'Text', props: {}, children: ['engine row'] }))
    const origin = { kind: 'plugin', name: 'live-vibe', asUser: true } as const
    const collapsed = await $.ui.mount({ plugin: 'live-vibe', surface, component: 'UserMessage',
      props: { text: prompt, origin, isExpanded: false } })
    expect((await collapsed.find({ type: 'Text' }))?.text)
      .toBe('front prompt (3 lines, ctrl+o to expand): task: Research end-of-turn detection.')

    const expanded = await $.ui.mount({ plugin: 'live-vibe', surface, component: 'UserMessage',
      props: { text: prompt, origin, isExpanded: true } })
    expect(await expanded.find({ text: /front prompt/ })).toBeUndefined()
    expect((await expanded.find({ type: 'Text' }))?.text).toBe('engine row')

    const typedRow = await $.ui.mount({ plugin: 'live-vibe', surface, component: 'UserMessage',
      props: { text: 'Fix the parser', origin, isExpanded: false } })
    expect(await typedRow.find({ text: /front prompt/ })).toBeUndefined()
  })
}

for (const surface of ['terminal', 'desktop'] as const) {
  test(`live vibe: a turn with nothing to add is one dim "no update" line (${surface})`, async ($, on) => {
    on('ui.render', () => ({ type: 'Text', props: {}, children: ['engine row'] }))
    for (const text of ['(nothing to add)', ' (Nothing to add.) ']) {
      const row = await $.ui.mount({ plugin: 'live-vibe', surface, component: 'AssistantMessage',
        props: { text, isFirstOfReply: true } })
      expect((await row.find({ type: 'Text' }))?.text).toBe('no update (nothing to add)')
    }
    const real = await $.ui.mount({ plugin: 'live-vibe', surface, component: 'AssistantMessage',
      props: { text: 'Nothing to add yet; the worker still runs.', isFirstOfReply: true } })
    expect((await real.find({ type: 'Text' }))?.text).toBe('engine row')
  })
}

// A worker's notification that repeats a report already given: Claude answers "(nothing to add)", whatever started
// the turn, and the front hears nothing; the relay prompt tells Claude so.
test("live vibe: a notification turn's '(nothing to add)' is not relayed, and the relay prompt asks for it", async ($, on) => {
  const posts: string[] = []
  const sidecar = fakeSidecar(4321)
  on('process.spawn', sidecar.spawn)
  on('turn.complete', (_$, e) => ({ text: e.answer }))
  on('http.fetch', (_$, e) => { posts.push(e.url); return ok })
  on('prompt.compose', () => ({ sections: [] }))

  await $.command.run({ command: 'livevibe', ...typed })
  await sidecar.isUp
  const relay = (await $.prompt.compose(compose(['Read', 'Agent']))).sections.find(x => x.id === 'live-vibe:relay')
  expect(relay?.text).toContain("a worker's notification that repeats what you already reported")
  expect(relay?.text).toContain('reply with exactly: (nothing to add)')
  await $.turn.complete({ answer: '(nothing to add)', durationMs: 1, isAborted: false, turnId: 'n1', reason: 'answer' })
  expect(posts).toEqual([])
  await $.command.run({ command: 'livevibe', ...typed })
  sidecar.release()
})
