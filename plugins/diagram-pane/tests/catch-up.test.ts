import { expect, mock, test } from 'claude-code/testing'

// A session with three turns of work and no map saved by this install: the
// infra session after moving diagram-pane to another marketplace
const MESSAGES = ['refactor the wireguard stack', 'fix the KICS finding', 'remove the tailnet'].flatMap(text => [
  { role: 'user', text, toolUses: [] },
  { role: 'assistant', text: 'done', toolUses: [] },
])

const MAP = { title: 'WireGuard access refactor', nodes: [{ id: 'ap', label: 'address-plan', kind: 'file' }, { id: 'acc', label: 'access/ stack', kind: 'component', now: true }], edges: [{ from: 'ap', to: 'acc' }] }
const USAGE = { input_tokens: 1, output_tokens: 1, cache_creation_input_tokens: 0, cache_read_input_tokens: 0 }

test('diagram_get draws the map at once for a session with work but no saved map', async ($, on) => {
  mock.store(on)
  mock.env(on, {})
  mock.clock(on)
  on('session.id', () => ({ value: 'cda0ce19' }))
  on('session.messages', () => ({ value: MESSAGES }))
  let forks = 0
  on('model.fork', () => (forks++, { value: { isAnswered: true, text: JSON.stringify(MAP), usage: USAGE } }) as never)

  const got = String(((await $.tool.call({ tool: 'mcp__diagram-pane__diagram_get' } as never)) as { result?: unknown }).result)
  expect(forks).toBe(1)
  expect(got).toContain('## map: "WireGuard access refactor"')
  expect(got).toContain('ap --> acc')
})
