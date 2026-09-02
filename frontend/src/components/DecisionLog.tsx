'use client'

import type { DecisionOut } from '@/api/types'
import { Empty, fmtTime } from '@/components/ui'

export function DecisionLog({ decisions }: { decisions: DecisionOut[] | undefined }) {
  if (!decisions?.length) return <Empty>No decisions logged yet.</Empty>
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead className="text-xs uppercase text-neutral-500">
          <tr>
            <th className="py-1 pr-3">when</th>
            <th className="py-1 pr-3">scene</th>
            <th className="py-1 pr-3">phase</th>
            <th className="py-1 pr-3">att.</th>
            <th className="py-1 pr-3">action</th>
            <th className="py-1 pr-3">score</th>
            <th className="py-1">reason</th>
          </tr>
        </thead>
        <tbody>
          {decisions.map((d, i) => (
            <tr key={i} className="border-t border-neutral-100">
              <td className="py-1 pr-3 whitespace-nowrap text-neutral-500">{fmtTime(d.created_at)}</td>
              <td className="py-1 pr-3">{d.scene_index ?? '—'}</td>
              <td className="py-1 pr-3">{d.phase}</td>
              <td className="py-1 pr-3">{d.attempt}</td>
              <td className={`py-1 pr-3 font-medium ${d.action === 'retry' ? 'text-amber-700' : d.action === 'approved' ? 'text-green-700' : ''}`}>{d.action}</td>
              <td className="py-1 pr-3">{d.score ?? ''}</td>
              <td className="py-1 text-neutral-700">{d.reason}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
