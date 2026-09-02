'use client'

import type { EnvReport } from '@/hooks/useSetup'

const DOT: Record<string, string> = { ok: 'text-green-600', warn: 'text-amber-600', fail: 'text-red-600' }

/** Compact list of failing/warning checks (the wizard's submit blocker). */
export function EnvSummary({ report, showOk = false }: { report: EnvReport | undefined; showOk?: boolean }) {
  if (!report) return null
  const rows = report.checks.filter((c) => showOk || c.status !== 'ok')
  if (rows.length === 0) return <p className="text-sm text-green-700">Environment OK.</p>
  return (
    <ul className="space-y-1 text-sm">
      {rows.map((c) => (
        <li key={c.name} className="flex gap-2">
          <span className={`font-mono text-xs ${DOT[c.status]}`}>{c.status.toUpperCase()}</span>
          <span className="font-medium">{c.name}</span>
          <span className="text-neutral-600">{c.detail}</span>
        </li>
      ))}
    </ul>
  )
}
