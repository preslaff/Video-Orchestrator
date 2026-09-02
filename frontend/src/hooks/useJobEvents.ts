'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { apiOrigin } from '@/api/client'
import { EVENT_TYPES, LIST_EVENTS, type JobEvent } from '@/api/events'
import { keys } from '@/api/keys'

export interface UseJobEventsOptions {
  enabled?: boolean
  /**
   * True while the job is running (or about to). A run segment ends with an
   * `end` event and the stream closes at an approval gate or at DONE; while
   * `active` stays true (or turns true again after an approve) the stream is
   * reopened from the last seen id, so the next segment's events arrive live.
   */
  active?: boolean
  /** Extra listener for every event (after the built-in query invalidation). */
  onEvent?: (e: JobEvent) => void
}

/**
 * Live progress for one job over SSE. Opens `/api/jobs/{id}/events?after=<last id>`
 * (`after=0` replays the ring buffer so a reload shows the timeline so far), closes
 * on `end` (otherwise the browser would reconnect forever), and lets the browser's
 * own reconnect resume from `Last-Event-ID`. Query caches are invalidated on events.
 */
export function useJobEvents(jobId: string | undefined, opts: UseJobEventsOptions = {}) {
  const { enabled = true, active = false, onEvent } = opts
  const qc = useQueryClient()
  const [events, setEvents] = useState<JobEvent[]>([])
  const [connected, setConnected] = useState(false)
  const [ended, setEnded] = useState(false)
  const [generation, setGeneration] = useState(0)
  const lastId = useRef(0)
  const endedRef = useRef(false)
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const onEventRef = useRef(onEvent)

  useEffect(() => {
    onEventRef.current = onEvent
  }, [onEvent])

  useEffect(() => {
    if (!jobId || !enabled || typeof EventSource === 'undefined') return
    // After an `end`, only reopen when the job is active again (a new run segment).
    if (endedRef.current && !active) return
    const url = `${apiOrigin()}/api/jobs/${jobId}/events?after=${lastId.current}`
    const es = new EventSource(url)

    const invalidateJob = () => {
      if (timer.current) clearTimeout(timer.current)
      timer.current = setTimeout(() => {
        void qc.invalidateQueries({ queryKey: keys.job(jobId) })
      }, 250)
    }

    const handle = (ev: MessageEvent) => {
      let data: JobEvent
      try {
        data = JSON.parse(ev.data) as JobEvent
      } catch {
        return
      }
      if (typeof data.id === 'number') lastId.current = Math.max(lastId.current, data.id)
      setEvents((prev) => [...prev, { ...data, _at: Date.now() }])
      invalidateJob()
      if (LIST_EVENTS.has(data.type)) void qc.invalidateQueries({ queryKey: ['jobs'] })
      if (data.type === 'end' || data.type === 'export' || data.type === 'error') {
        void qc.invalidateQueries({ queryKey: keys.decisions(jobId) })
        void qc.invalidateQueries({ queryKey: keys.provenance(jobId) })
        void qc.invalidateQueries({ queryKey: keys.exportStatus(jobId) })
      }
      onEventRef.current?.(data)
      if (data.type === 'end') {
        es.close()
        endedRef.current = true
        setEnded(true)
        setConnected(false)
      }
    }

    for (const name of EVENT_TYPES) es.addEventListener(name, handle as EventListener)
    es.onopen = () => {
      endedRef.current = false
      setEnded(false)
      setConnected(true)
    }
    es.onerror = () => setConnected(false)

    return () => {
      for (const name of EVENT_TYPES) es.removeEventListener(name, handle as EventListener)
      es.close()
      if (timer.current) clearTimeout(timer.current)
    }
  }, [jobId, enabled, active, generation, qc])

  const reconnect = useCallback(() => {
    endedRef.current = false
    setEnded(false)
    setGeneration((g) => g + 1)
  }, [])

  return { events, connected, ended, reconnect }
}
