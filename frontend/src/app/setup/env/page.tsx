'use client'

import { useState } from 'react'
import { errorMessage } from '@/api/client'
import { useConfig, useEnv, useProfiles, useRecheckEnv } from '@/hooks/useSetup'
import { Button, Card, Empty, ErrorBox, fmtTime } from '@/components/ui'

const STYLE: Record<string, string> = { ok: 'bg-green-100 text-green-800', warn: 'bg-amber-100 text-amber-900', fail: 'bg-red-100 text-red-800' }

export default function EnvironmentPage() {
  const [profileId, setProfileId] = useState('')
  const [deep, setDeep] = useState(false)
  const profiles = useProfiles()
  const env = useEnv(profileId || undefined, deep)
  const recheck = useRecheckEnv(profileId || undefined, deep)
  const cfg = useConfig()
  const report = recheck.data ?? env.data

  return (
    <div className="grid gap-4 lg:grid-cols-3">
      <Card className="lg:col-span-2" title="Environment check" actions={
        <div className="flex items-center gap-2 text-sm">
          <select value={profileId} onChange={(e) => setProfileId(e.target.value)} className="rounded border border-neutral-300 bg-white px-2 py-1">
            <option value="">any profile</option>
            {profiles.data?.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
          </select>
          <label className="flex items-center gap-1"><input type="checkbox" checked={deep} onChange={(e) => setDeep(e.target.checked)} /> deep (vision probe)</label>
          <Button variant="primary" disabled={recheck.isPending} onClick={() => recheck.mutate()}>{recheck.isPending ? 'Checking…' : 'Re-run'}</Button>
        </div>
      }>
        <ErrorBox message={env.error ? errorMessage(env.error) : recheck.error ? errorMessage(recheck.error) : null} />
        {!report && env.isLoading && <Empty>Checking…</Empty>}
        {report && (
          <>
            <p className="mb-2 text-sm text-neutral-600">
              overall <span className={`rounded px-2 py-0.5 text-xs font-medium ${STYLE[report.status]}`}>{report.status}</span> · {fmtTime(report.checked_at)} · {report.duration_ms} ms
              {!report.ok && <span className="ml-2 text-red-700">— job submits are blocked until the FAIL rows are fixed (or forced).</span>}
            </p>
            <table className="w-full text-left text-sm">
              <thead className="text-xs uppercase text-neutral-500"><tr><th className="py-1 pr-3">status</th><th className="py-1 pr-3">check</th><th className="py-1">detail</th></tr></thead>
              <tbody>
                {report.checks.map((c) => (
                  <tr key={c.name} className="border-t border-neutral-100 align-top">
                    <td className="py-1 pr-3"><span className={`rounded px-2 py-0.5 text-xs font-medium ${STYLE[c.status]}`}>{c.status}</span></td>
                    <td className="py-1 pr-3 font-medium whitespace-nowrap">{c.name}{c.data?.used === true ? <span className="ml-1 text-xs font-normal text-neutral-500">(used)</span> : null}</td>
                    <td className="py-1 text-neutral-700">{c.detail}{Array.isArray(c.data?.problems) && (
                      <ul className="mt-1 list-inside list-disc text-xs text-neutral-600">{(c.data.problems as string[]).map((p) => <li key={p}>{p}</li>)}</ul>
                    )}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}
      </Card>
      <Card title="Effective configuration">
        {cfg.data ? (
          <dl className="space-y-1 text-xs">
            {Object.entries(cfg.data).map(([k, v]) => (
              <div key={k} className="grid grid-cols-[9rem_1fr] gap-2"><dt className="text-neutral-500">{k}</dt><dd className="break-all font-mono">{String(v)}</dd></div>
            ))}
          </dl>
        ) : <Empty>Loading…</Empty>}
      </Card>
    </div>
  )
}
