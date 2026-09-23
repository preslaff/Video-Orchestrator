'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { apiOrigin } from '@/api/client'
import { DELIVERABLE_EVENTS, EVENT_TYPES, LIST_EVENTS, type JobEvent } from '@/api/events'
import { keys } from '@/api/keys'

/**
 * The timeline keeps the most recent events and drops the rest.
 *
 * It only ever grew before. A job page left open across many runs — regenerate a scene, lock
 * a label, re-animate, a dozen times over an evening — accumulated every event of every
 * segment, and the progress panel rescans the list once a second while a job runs. Nothing
 * looked wrong at first and it got steadily worse, until a reload cured it. Well past what
 * any panel shows, and the persisted job (not this list) is what the page really renders from.
 */
const MAX_EVENTS = 600

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
  // Where this connection's replay stops. A job's ring buffer holds every run segment, so a
  // replay can carry the `end` of an earlier one; stopping there used to leave the page blind
  // to the segment now running (a regenerated keyframe appeared only after a manual reload).
  const watermark = useRef(0)
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
    watermark.current = 0
    let replayEndsHere = 0        // set by the `open` event; 0 until then
    let liveSegment = false

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
      if (data.type === 'open') {
        replayEndsHere = typeof data.last_id === 'number' ? data.last_id : 0
        watermark.current = replayEndsHere
        liveSegment = data.active === true
        return                     // a protocol marker, not something to show or invalidate
      }
      if (typeof data.id === 'number') lastId.current = Math.max(lastId.current, data.id)
      setEvents((prev) => {
        const next = prev.length >= MAX_EVENTS ? prev.slice(prev.length - MAX_EVENTS + 1) : prev.slice()
        next.push({ ...data, _at: Date.now() })
        return next
      })
      invalidateJob()
      if (LIST_EVENTS.has(data.type)) void qc.invalidateQueries({ queryKey: ['jobs'] })
      if (DELIVERABLE_EVENTS.has(data.type)) {
        void qc.invalidateQueries({ queryKey: keys.decisions(jobId) })
        void qc.invalidateQueries({ queryKey: keys.provenance(jobId) })
        void qc.invalidateQueries({ queryKey: keys.exportStatus(jobId) })
        void qc.invalidateQueries({ queryKey: keys.assembly(jobId) })
      }
      onEventRef.current?.(data)
      // Mirror the server's rule: an `end` from the replay of an earlier segment is history
      // while a segment is running — the stream (and this page) stays live for the real one.
      const historical = liveSegment && !data.final && (data.id ?? 0) <= replayEndsHere
      if (data.type === 'end' && !historical) {
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
