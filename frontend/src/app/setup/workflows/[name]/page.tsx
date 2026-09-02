'use client'

import { use } from 'react'
import Link from 'next/link'
import { errorMessage } from '@/api/client'
import { useWorkflow } from '@/hooks/useSetup'
import { Card, Empty, ErrorBox } from '@/components/ui'

export default function WorkflowPage({ params }: { params: Promise<{ name: string }> }) {
  const { name } = use(params)
  const wf = useWorkflow(name)
  if (wf.isLoading) return <Empty>Loading…</Empty>
  if (wf.error || !wf.data) return <ErrorBox message={errorMessage(wf.error, 'Workflow not found')} />
  const w = wf.data
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <Link href="/setup/workflows" className="text-sm text-neutral-500 hover:underline">← workflows</Link>
        <h2 className="text-lg font-semibold">{w.name}</h2>
        <span className="rounded bg-neutral-200 px-2 py-0.5 text-xs">{w.kind}</span>
        <span className="text-xs text-neutral-500">{w.builtin ? 'built-in' : `user · ${w.source}`}</span>
      </div>
      {w.description && <p className="text-sm text-neutral-700">{w.description}</p>}
      {w.problems && w.problems.length > 0 && <ErrorBox message={w.problems.join(' · ')} />}
      <div className="grid gap-4 lg:grid-cols-3">
        <Card title="Parameters" className="lg:col-span-2">
          <table className="w-full text-left text-sm">
            <thead className="text-xs uppercase text-neutral-500"><tr><th className="py-1 pr-3">param</th><th className="py-1 pr-3">node.field</th><th className="py-1 pr-3">type</th><th className="py-1 pr-3">constraint</th><th className="py-1">required</th></tr></thead>
            <tbody>
              {Object.entries(w.params).map(([n, p]) => (
                <tr key={n} className="border-t border-neutral-100">
                  <td className="py-1 pr-3 font-medium">{n}{w.inputs[n] ? <span className="ml-1 text-xs text-neutral-500">(input:{w.inputs[n]})</span> : null}</td>
                  <td className="py-1 pr-3 font-mono text-xs">{p.targets.map((t) => `${t.node}.${t.field}`).join(', ')}</td>
                  <td className="py-1 pr-3">{p.type}{p.enum_from ? <span className="text-xs text-neutral-500"> ← {p.enum_from.class_type}.{p.enum_from.input}</span> : null}</td>
                  <td className="py-1 pr-3 text-xs">{[p.constraint, p.min != null ? `≥${p.min}` : null, p.max != null ? `≤${p.max}` : null].filter(Boolean).join(' ') || '—'}</td>
                  <td className="py-1">{p.required ? 'yes' : ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
        <div className="space-y-4">
          <Card title="Template">
            <dl className="space-y-1 text-sm">
              <div><dt className="text-xs text-neutral-500">file</dt><dd className="font-mono text-xs">{w.template}</dd></div>
              <div><dt className="text-xs text-neutral-500">output</dt><dd>{w.outputs.type ?? '?'} from node(s) {w.outputs.nodes.join(', ') || 'any'}</dd></div>
              <div><dt className="text-xs text-neutral-500">timeout</dt><dd>{w.timeout ? `${w.timeout}s` : 'default'}</dd></div>
              {w.tags.length > 0 && <div><dt className="text-xs text-neutral-500">tags</dt><dd>{w.tags.join(', ')}</dd></div>}
            </dl>
          </Card>
          <Card title="Requires">
            <div className="text-xs"><span className="text-neutral-500">node classes:</span> {w.requires.nodes.join(', ')}</div>
            {w.requires.models.length > 0 && (
              <ul className="mt-2 space-y-0.5 text-xs">{w.requires.models.map((m, i) => <li key={i}><span className="text-neutral-500">{m.class_type}.{m.field}</span> = <span className="font-mono">{m.value}</span></li>)}</ul>
            )}
          </Card>
        </div>
      </div>
    </div>
  )
}
