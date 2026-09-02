'use client'

import { describeEvent, type JobEvent } from '@/api/events'
import type { DecisionOut } from '@/api/types'
import { Empty } from '@/components/ui'

/** Live SSE events + persisted decisions, oldest first. */
export function Timeline({ events, decisions, connected, ended }: { events: JobEvent[]; decisions?: DecisionOut[]; connected: boolean; ended: boolean }) {
  return (
    <div className="space-y-2">
      <div className="text-xs text-neutral-500">
        {ended ? 'stream closed' : connected ? '● live' : '○ connecting…'} · {events.length} event(s)
      </div>
      {events.length === 0 && !decisions?.length && <Empty>Nothing yet.</Empty>}
      <ol className="max-h-72 space-y-1 overflow-auto text-sm">
        {events.map((e, i) => (
          <li key={`${e.id ?? 'x'}-${i}`} className={`flex gap-2 ${e.type === 'error' ? 'text-red-700' : e.type === 'gate' ? 'text-amber-800' : ''}`}>
            <span className="w-8 shrink-0 text-right font-mono text-xs text-neutral-400">{e.id ?? ''}</span>
            <span>{describeEvent(e)}</span>
          </li>
        ))}
      </ol>
    </div>
  )
}
