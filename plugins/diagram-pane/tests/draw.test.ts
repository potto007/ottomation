import { expect, test } from 'claude-code/testing'
import { findDiagrams, parseMermaid, drawGraph } from '../hooks/draw.js'

const reply = 'The flow:\n\n```mermaid\nflowchart TD\n  A[Cart] -->|total| B{Over 200?}\n  B -- yes --> C[Discount]\n  B -- no --> D[Pay]\n  C --> D\n```\n\n```bash\nls -> out\n```\n'

test('finds the mermaid block and skips a bash block', async () => {
  const found = findDiagrams(reply)
  expect(found.length).toBe(1)
  expect(found[0].kind).toBe('mermaid')
  expect(found[0].title).toBe('The flow')
})

test('finds an ASCII box diagram in a plain code block', async () => {
  const found = findDiagrams('```\n┌───┐   ┌───┐\n│ a │──▶│ b │\n└───┘   └───┘\n```')
  expect(found.length).toBe(1)
  expect(found[0].kind).toBe('ascii')
})

test('parses nodes, shapes and edge labels', async () => {
  const g = parseMermaid(findDiagrams(reply)[0].code)!
  expect(g.nodes.map(n => n.label)).toEqual(['Cart', 'Over 200?', 'Discount', 'Pay'])
  expect(g.nodes[1].box).toBe('double')
  expect(g.edges.map(e => e.label)).toEqual(['total', 'yes', 'no', ''])
})

test('draws boxes and arrows inside the width', async () => {
  const lines = drawGraph(parseMermaid(findDiagrams(reply)[0].code)!, 40)
  expect(lines.every(l => l.length <= 40)).toBe(true)
  expect(lines.join('\n')).toContain('│ Cart │')
  expect(lines.join('\n')).toContain('▼ yes')
})
