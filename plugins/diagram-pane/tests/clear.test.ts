import { expect, mock, test } from 'claude-code/testing'

// /clear keeps the process, so session.start does not fire: only the session
// id and the transcript change
const PIN = 'The flow:\n\n```mermaid\nflowchart TD\n  A[Cart] --> B[Pay]\n```\n'
const A = [
  { role: 'user', text: 'explain checkout', toolUses: [] },
  { role: 'assistant', text: PIN, toolUses: [] },
  { role: 'user', text: 'now the infra', toolUses: [] },
  { role: 'assistant', text: 'ok', toolUses: [] },
]
const B = [
  { role: 'user', text: 'something else', toolUses: [] },
  { role: 'assistant', text: 'ok', toolUses: [] },
]
const MAP = { title: 'late map', nodes: [{ id: 'x', label: 'old work', kind: 'file' }], edges: [] }
const USAGE = { input_tokens: 1, output_tokens: 1, cache_creation_input_tokens: 0, cache_read_input_tokens: 0 }

const get = async ($: any) => String(((await $.tool.call({ tool: 'mcp__diagram-pane__diagram_get' } as never)) as { result?: unknown }).result)

test('a new session id starts the pane fresh, and a fork from the old session cannot land in it', async ($, on) => {
  mock.store(on)
  mock.env(on, {})
  mock.clock(on)
  on('ui.panes', () => ({ value: [{ id: 'diagrams', title: 'Diagrams', isPlaced: true }] }))
  let sid = 'A'
  let messages: unknown[] = A
  on('session.id', () => ({ value: sid }))
  on('session.messages', () => ({ value: messages }))
  let release = () => {}
  const held = new Promise<void>(r => (release = r))
  let started = () => {}
  const forking = new Promise<void>(r => (started = r))
  let forks = 0
  on('model.fork', async () => {
    if (forks++) return { value: { isAnswered: false, reason: 'api-error', usage: USAGE } } as never
    started()
    await held
    return { value: { isAnswered: true, text: JSON.stringify(MAP), usage: USAGE } } as never
  })

  // Session A: a pin and a hand-drawn map
  await $.tool.call({ tool: 'mcp__diagram-pane__diagram_set', target: 'map', mermaid: 'flowchart TD\n  s[stack.tf] --> v[vpc.tf]', title: 'infra' } as never)
  const before = await get($)
  expect(before).toContain('## map: "infra"')
  expect(before).toContain('## pin:1: "The flow"')

  // /clear: session B
  sid = 'B'
  messages = B
  const after = await get($)
  expect(after).toContain('## map: not drawn yet')
  expect(after).toContain('No pinned diagrams.')
  expect(after).not.toContain('infra')

  // Session C has work but no map, so its catch-up fork starts; /clear again
  // (session D) before the fork answers
  await $.tool.call({ tool: 'mcp__diagram-pane__diagram_set', target: 'map', mermaid: 'flowchart TD\n  b[b.tf]', title: 'b map' } as never)
  sid = 'C'
  messages = [...A, ...A]
  const late = get($)
  await forking
  sid = 'D'
  messages = B
  await get($)
  release()
  await late
  const d = await get($)
  expect(d).toContain('## map: not drawn yet')
  expect(d).not.toContain('late map')

  // Each session kept its own saved map, and C did not get the late one
  const visit = async (id: string, msgs: unknown[]) => ((sid = id), (messages = msgs), get($))
  expect(await visit('A', A)).toContain('## map: "infra"')
  expect(await visit('B', B)).toContain('## map: "b map"')
  const c = await visit('C', [...A, ...A])
  expect(c).toContain('## map: not drawn yet')
  expect(c).not.toContain('late map')
})
