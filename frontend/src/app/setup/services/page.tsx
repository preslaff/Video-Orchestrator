'use client'

import { useState } from 'react'
import { errorMessage } from '@/api/client'
import { useConfig } from '@/hooks/useSetup'
import { useServiceLogs, useServices, type GpuHolder, type ServiceState, type ServicesReport } from '@/hooks/useServices'
import { Button, Card, Empty, ErrorBox } from '@/components/ui'

// Same vocabulary as the environment page, plus `down` — which is not a fault.
const STYLE: Record<string, string> = {
  ok: 'bg-green-100 text-green-800',
  warn: 'bg-amber-100 text-amber-900',
  fail: 'bg-red-100 text-red-800',
  down: 'bg-neutral-200 text-neutral-700',
}

const HOLDER_COLOUR = ['bg-sky-500', 'bg-violet-500', 'bg-amber-500', 'bg-rose-500', 'bg-teal-500']

function duration(sec: number | null): string {
  if (sec == null) return '—'
  if (sec < 60) return `${Math.round(sec)}s`
  if (sec < 3600) return `${Math.round(sec / 60)}m`
  if (sec < 86400) return `${Math.floor(sec / 3600)}h${String(Math.floor((sec % 3600) / 60)).padStart(2, '0')}m`
  return `${Math.floor(sec / 86400)}d${String(Math.floor((sec % 86400) / 3600)).padStart(2, '0')}h`
}

/** The single most useful thing to look at before submitting: who has the card. */
function GpuBar({ gpu }: { gpu: ServicesReport['gpu'] }) {
  if (!gpu) return <Empty>No GPU visible (nvidia-smi unavailable) — VRAM is not being arbitrated.</Empty>
  const pct = (mb: number) => `${Math.max(0, (100 * mb) / gpu.total)}%`
  return (
    <div>
      <div className="mb-2 flex h-6 w-full overflow-hidden rounded bg-neutral-100" data-testid="gpu-bar">
        {gpu.holders.map((h: GpuHolder, i: number) => (
          <div key={h.pid} className={`${HOLDER_COLOUR[i % HOLDER_COLOUR.length]} ${h.evictable ? '' : 'opacity-60'}`}
               style={{ width: pct(h.used_mb) }}
               title={`${h.label} — pid ${h.pid}, ${h.used_mb} MB, up ${h.uptime}${h.evictable ? '' : ' (fixed cost)'}`} />
        ))}
        {gpu.unattributed_mb > 0 && (
          <div className="bg-neutral-400" style={{ width: pct(gpu.unattributed_mb) }} title={`driver context and overhead — ${gpu.unattributed_mb} MB`} />
        )}
      </div>
      <p className="text-sm text-neutral-700">
        <span className="font-medium">{gpu.free} MB free</span> of {gpu.total}
        {gpu.holders.length > 0 && ' — '}
        {gpu.holders.map((h: GpuHolder, i: number) => (
          <span key={h.pid}>
            {i > 0 && ', '}
            <span className={`inline-block h-2 w-2 rounded-sm ${HOLDER_COLOUR[i % HOLDER_COLOUR.length]}`} />{' '}
            {h.label} {h.used_mb} MB
            {h.service === null && <span className="ml-1 text-xs text-amber-800">(unknown — vorch will not touch it)</span>}
          </span>
        ))}
      </p>
    </div>
  )
}

function Row({ s, onLogs }: { s: ServiceState; onLogs: () => void }) {
  return (
    <tr className="border-t border-neutral-100 align-top">
      <td className="py-1.5 pr-3"><span className={`rounded px-2 py-0.5 text-xs font-medium ${STYLE[s.status]}`}>{s.status}</span></td>
      <td className="py-1.5 pr-3 whitespace-nowrap font-medium">
        {s.name}
        {s.manage === 'never' && <span className="ml-1 text-xs font-normal text-neutral-500">(not managed)</span>}
        {s.manage === 'adopt' && <span className="ml-1 text-xs font-normal text-neutral-500">(monitored)</span>}
        {s.lifetime !== 'long' && <span className="ml-1 rounded bg-sky-100 px-1 text-xs font-normal text-sky-800">{s.lifetime}</span>}
      </td>
      <td className="py-1.5 pr-3 whitespace-nowrap text-neutral-600">{s.vram_mb ? `${s.vram_mb} MB` : '—'}</td>
      <td className="py-1.5 pr-3 whitespace-nowrap text-neutral-600">{duration(s.uptime_sec)}</td>
      <td className="py-1.5 text-neutral-700">
        {s.detail || s.description}
        {s.orphan && <p className="mt-0.5 text-xs text-red-700">Left behind by a vorch that is gone. It will keep holding the card until it is stopped.</p>}
        {s.identity === 'mismatch' && <p className="mt-0.5 text-xs text-red-700">This is not the instance the manifest describes — results from it would be wrong.</p>}
        <button type="button" onClick={onLogs} className="mt-0.5 text-xs text-sky-700 underline">log</button>
      </td>
    </tr>
  )
}

export default function ServicesPage() {
  const [logFor, setLogFor] = useState<string | null>(null)
  const report = useServices()
  const logs = useServiceLogs(logFor)
  const cfg = useConfig()

  return (
    <div className="space-y-4">
      <Card title="GPU">
        <ErrorBox message={report.error ? errorMessage(report.error) : null} />
        {report.data ? <GpuBar gpu={report.data.gpu} /> : <Empty>Looking…</Empty>}
      </Card>

      <Card title="Services" actions={
        <Button disabled={report.isFetching} onClick={() => void report.refetch()}>{report.isFetching ? 'Checking…' : 'Refresh'}</Button>
      }>
        {report.data && report.data.problems.length > 0 && (
          <ul className="mb-3 list-inside list-disc rounded bg-red-50 p-2 text-sm text-red-800">
            {report.data.problems.map((p) => <li key={p}>{p}</li>)}
          </ul>
        )}
        {!report.data ? <Empty>Looking…</Empty> : (
          <table className="w-full text-left text-sm">
            <thead className="text-xs uppercase text-neutral-500">
              <tr><th className="py-1 pr-3">status</th><th className="py-1 pr-3">service</th><th className="py-1 pr-3">vram</th><th className="py-1 pr-3">uptime</th><th className="py-1">detail</th></tr>
            </thead>
            <tbody>
              {report.data.services.map((s) => <Row key={s.name} s={s} onLogs={() => setLogFor(s.name === logFor ? null : s.name)} />)}
            </tbody>
          </table>
        )}
        <p className="mt-3 text-xs text-neutral-500">
          <code>down</code> is not a fault: a service started on demand is meant to be absent until it is needed.
          A service you launched yourself shows as <em>started outside vorch</em> — that is supported, and vorch
          will not stop it. Manifest: <code>{report.data?.services_file ?? cfg.data?.services_file ?? '—'}</code>
        </p>
      </Card>

      {logFor && (
        <Card title={`Log — ${logFor}`} actions={<Button onClick={() => setLogFor(null)}>Close</Button>}>
          {logs.isLoading ? <Empty>Reading…</Empty> : (
            <pre className="max-h-96 overflow-auto rounded bg-neutral-900 p-3 text-xs text-neutral-100">{logs.data?.text}</pre>
          )}
        </Card>
      )}
    </div>
  )
}
