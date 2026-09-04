'use client'

import Link from 'next/link'
import { errorMessage } from '@/api/client'
import { useWorkflows } from '@/hooks/useSetup'
import { Card, Empty, ErrorBox } from '@/components/ui'

export default function WorkflowsPage() {
  const wf = useWorkflows()
  return (
    <div className="space-y-4">
      <Card title="Registered ComfyUI workflows">
        <ErrorBox message={wf.error ? errorMessage(wf.error) : null} />
        {wf.isLoading && <Empty>Loading…</Empty>}
        {wf.data && (
          <table className="w-full text-left text-sm">
            <thead className="text-xs uppercase text-neutral-500"><tr><th className="py-1 pr-3">name</th><th className="py-1 pr-3">kind</th><th className="py-1 pr-3">template</th><th className="py-1 pr-3">inputs</th><th className="py-1 pr-3">output</th><th className="py-1 pr-3">backend</th><th className="py-1">source</th></tr></thead>
            <tbody>
              {wf.data.workflows.map((w) => (
                <tr key={w.name} className="border-t border-neutral-100">
                  <td className="py-1 pr-3"><Link href={`/setup/workflows/${w.name}`} className="font-medium hover:underline">{w.name}</Link></td>
                  <td className="py-1 pr-3">{w.kind}</td>
                  <td className="py-1 pr-3 font-mono text-xs">{w.template}</td>
                  <td className="py-1 pr-3 text-xs">{Object.entries(w.inputs).map(([k, v]) => `${k}:${v}`).join(', ') || '—'}</td>
                  <td className="py-1 pr-3">{w.outputs.type ?? '?'}</td>
                  <td className="py-1 pr-3 text-xs">{w.backend === 'local' ? <span className="text-neutral-500">local</span> : <span className="rounded bg-sky-100 px-1.5 py-0.5 font-medium text-sky-800" title="remote ComfyUI (see docs/modal-backend.md)">{w.backend}</span>}</td>
                  <td className="py-1 text-xs text-neutral-500">{w.builtin ? 'built-in' : 'user'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {wf.data && Object.keys(wf.data.errors).length > 0 && (
          <div className="mt-3 rounded border border-red-200 bg-red-50 p-2 text-sm">
            <div className="font-medium text-red-800">Invalid user workflows (skipped)</div>
            <ul className="list-inside list-disc text-red-700">{Object.entries(wf.data.errors).map(([n, e]) => <li key={n}><b>{n}</b>: {e}</li>)}</ul>
          </div>
        )}
      </Card>
      <Card title="Add a workflow (no Python needed)">
        <ol className="list-inside list-decimal space-y-1 text-sm text-neutral-700">
          <li>In ComfyUI: <b>Export (API)</b> the graph as JSON.</li>
          <li><code className="rounded bg-neutral-100 px-1">python -m vorch registry inspect &lt;file.json&gt; --online</code> — see nodes + role guesses.</li>
          <li><code className="rounded bg-neutral-100 px-1">python -m vorch registry add &lt;file.json&gt; --kind postprocess</code> — scaffolds the YAML into <code className="rounded bg-neutral-100 px-1">{wf.data?.user_dir ?? '~/vorch/workflows'}</code>.</li>
          <li>Edit names/description, then <code className="rounded bg-neutral-100 px-1">python -m vorch registry validate &lt;name&gt; --online</code>.</li>
          <li>Reference it from a profile (keyframe/video/face-swap workflow or a post-step) — it shows up here after a reload.</li>
          <li>Optional: <code className="rounded bg-neutral-100 px-1">backend: &lt;name&gt;</code> in the YAML runs it on a remote ComfyUI from <code className="rounded bg-neutral-100 px-1">~/vorch/backends.yaml</code> (e.g. Modal); outputs are downloaded into the local layout.</li>
        </ol>
      </Card>
    </div>
  )
}
