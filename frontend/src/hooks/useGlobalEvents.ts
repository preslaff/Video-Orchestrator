'use client'

import { useEffect, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { apiOrigin } from '@/api/client'
import { EVENT_TYPES, LIST_EVENTS, type JobEvent } from '@/api/events'

/** Every job's events (for the jobs list): refreshes the list on status changes. */
export function useGlobalEvents(enabled = true) {
  const qc = useQueryClient()
  const [connected, setConnected] = useState(false)
  const [last, setLast] = useState<JobEvent | null>(null)

  useEffect(() => {
    if (!enabled || typeof EventSource === 'undefined') return
    const es = new EventSource(`${apiOrigin()}/api/events`)
    const handle = (ev: MessageEvent) => {
      try {
        const data = JSON.parse(ev.data) as JobEvent
        setLast(data)
        if (LIST_EVENTS.has(data.type)) void qc.invalidateQueries({ queryKey: ['jobs'] })
        if (data.job_id) void qc.invalidateQueries({ queryKey: ['job', data.job_id] })
      } catch {
        /* ignore malformed */
      }
    }
    for (const name of EVENT_TYPES) es.addEventListener(name, handle as EventListener)
    es.onopen = () => setConnected(true)
    es.onerror = () => setConnected(false)
    return () => es.close()
  }, [enabled, qc])

  return { connected, last }
}
