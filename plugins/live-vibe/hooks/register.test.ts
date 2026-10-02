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

test('/livevibe model lists what the front server serves, and /livevibe model foo sets the field', async ($, on) => {
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
