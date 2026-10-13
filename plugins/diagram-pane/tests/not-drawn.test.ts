import { expect, mock, test } from 'claude-code/testing'

const MESSAGES = ['one', 'two', 'three'].flatMap(text => [
  { role: 'user', text, toolUses: [] },
  { role: 'assistant', text: 'ok', toolUses: [] },
])
const USAGE = { input_tokens: 1, output_tokens: 1, cache_creation_input_tokens: 0, cache_read_input_tokens: 0 }

test('with no map, diagram_get says why and when the next check runs, and tries the fork once', async ($, on) => {
  mock.store(on)
  mock.env(on, {})
  mock.clock(on)
  on('session.id', () => ({ value: 's2' }))
  on('session.messages', () => ({ value: MESSAGES }))
  let forks = 0
  on('model.fork', () => (forks++, { value: { isAnswered: false, reason: 'api-error', status: 529, error: 'overloaded_error', usage: USAGE } }) as never)

  const get = async () => String(((await $.tool.call({ tool: 'mcp__diagram-pane__diagram_get' } as never)) as { result?: unknown }).result)
  const first = await get()
  expect(first).toContain('## map: not drawn yet. Last status: fork failed: api-error.')
  expect(first).toContain('maps are kept per install')
  expect(first).toContain('/diagram to draw it now')
  await get()
  expect(forks).toBe(1)
})
