import { act, renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useJobEvents } from './useJobEvents'

class FakeEventSource {
  static instances: FakeEventSource[] = []
  url: string
  listeners = new Map<string, EventListener[]>()
  closed = false
  onopen: (() => void) | null = null
  onerror: (() => void) | null = null
  constructor(url: string) {
    this.url = url
    FakeEventSource.instances.push(this)
  }
  addEventListener(name: string, fn: EventListener) {
    this.listeners.set(name, [...(this.listeners.get(name) ?? []), fn])
  }
  removeEventListener(name: string, fn: EventListener) {
    this.listeners.set(name, (this.listeners.get(name) ?? []).filter((f) => f !== fn))
  }
  close() {
    this.closed = true
  }
  emit(type: string, payload: Record<string, unknown>) {
    const ev = { data: JSON.stringify({ type, ...payload }) } as MessageEvent
    for (const fn of this.listeners.get(type) ?? []) fn(ev)
  }
}

describe('useJobEvents', () => {
  let qc: QueryClient
  const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={qc}>{children}</QueryClientProvider>

  beforeEach(() => {
    FakeEventSource.instances = []
    vi.stubGlobal('EventSource', FakeEventSource)
    qc = new QueryClient()
  })
  afterEach(() => vi.unstubAllGlobals())

  it('subscribes from after=0, tracks ids, invalidates the job and closes on end', async () => {
    const invalidate = vi.spyOn(qc, 'invalidateQueries')
    const onEvent = vi.fn()
    const { result } = renderHook(() => useJobEvents('j1', { onEvent }), { wrapper })
    const es = FakeEventSource.instances[0]
    expect(es.url).toBe('/api/jobs/j1/events?after=0')
    expect([...es.listeners.keys()]).toEqual(expect.arrayContaining(['status', 'keyframe', 'gate', 'end']))

    act(() => es.onopen?.())
    expect(result.current.connected).toBe(true)

    act(() => es.emit('status', { id: 1, job_id: 'j1', status: 'running' }))
    act(() => es.emit('keyframe', { id: 2, job_id: 'j1', index: 0, path: '/k.png' }))
    expect(result.current.events.map((e) => e.type)).toEqual(['status', 'keyframe'])
    expect(onEvent).toHaveBeenCalledTimes(2)
    await waitFor(() => expect(invalidate).toHaveBeenCalledWith({ queryKey: ['job', 'j1'] }))
    expect(invalidate).toHaveBeenCalledWith({ queryKey: ['jobs'] })

    act(() => es.emit('end', { id: 3, job_id: 'j1' }))
    expect(es.closed).toBe(true)
    expect(result.current.ended).toBe(true)
    expect(result.current.connected).toBe(false)
  })

  it('reconnects from the last seen id', () => {
    const { result } = renderHook(() => useJobEvents('j2'), { wrapper })
    const first = FakeEventSource.instances[0]
    act(() => first.emit('status', { id: 7, job_id: 'j2', status: 'running' }))
    act(() => result.current.reconnect())
    expect(first.closed).toBe(true)
    expect(FakeEventSource.instances[1].url).toBe('/api/jobs/j2/events?after=7')
  })

  it('resumes across run segments while active, stays closed while paused at a gate', () => {
    const { result, rerender } = renderHook(({ active }: { active: boolean }) => useJobEvents('j4', { active }), {
      wrapper,
      initialProps: { active: true },
    })
    const first = FakeEventSource.instances[0]
    act(() => first.onopen?.())
    act(() => first.emit('status', { id: 1, job_id: 'j4', status: 'running' }))
    act(() => first.emit('gate', { id: 2, job_id: 'j4', gate: 'storyboard', status: 'needs_approval' }))
    act(() => first.emit('end', { id: 3, job_id: 'j4' }))
    expect(first.closed).toBe(true)
    expect(result.current.ended).toBe(true)

    // The job is paused at the gate: no stream while it is not active.
    rerender({ active: false })
    expect(FakeEventSource.instances).toHaveLength(1)

    // Approved -> running again: reopen from the last seen id and keep receiving.
    rerender({ active: true })
    expect(FakeEventSource.instances).toHaveLength(2)
    const second = FakeEventSource.instances[1]
    expect(second.url).toBe('/api/jobs/j4/events?after=3')
    act(() => second.onopen?.())
    expect(result.current.ended).toBe(false)
    act(() => second.emit('keyframe', { id: 4, job_id: 'j4', index: 0, path: '/k.png' }))
    expect(result.current.events.map((e) => e.type)).toEqual(['status', 'gate', 'end', 'keyframe'])

    // DONE: end again, job no longer active -> stays closed.
    act(() => second.emit('end', { id: 5, job_id: 'j4' }))
    rerender({ active: false })
    expect(FakeEventSource.instances).toHaveLength(2)
  })

  it('does nothing when disabled or without a job id', () => {
    renderHook(() => useJobEvents(undefined), { wrapper })
    renderHook(() => useJobEvents('j3', { enabled: false }), { wrapper })
    expect(FakeEventSource.instances).toHaveLength(0)
  })
})

describe('useJobEvents across run segments', () => {
  let qc: QueryClient
  const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={qc}>{children}</QueryClientProvider>

  beforeEach(() => {
    FakeEventSource.instances = []
    vi.stubGlobal('EventSource', FakeEventSource)
    qc = new QueryClient()
  })
  afterEach(() => vi.unstubAllGlobals())

  it('keeps listening past a replayed end while a segment is running', async () => {
    const invalidate = vi.spyOn(qc, 'invalidateQueries')
    const { result } = renderHook(() => useJobEvents('j1', { active: true }), { wrapper })
    const es = FakeEventSource.instances[0]
    act(() => es.onopen?.())
    // The buffer holds an earlier segment: open says the replay ends at id 3 and a run is live.
    act(() => es.emit('open', { job_id: 'j1', last_id: 3, active: true, status: 'running' }))
    act(() => es.emit('keyframe', { id: 2, job_id: 'j1', index: 0 }))
    act(() => es.emit('end', { id: 3, job_id: 'j1' }))
    expect(es.closed).toBe(false)                       // history, not this segment's end
    expect(result.current.ended).toBe(false)

    invalidate.mockClear()
    act(() => es.emit('keyframe', { id: 4, job_id: 'j1', index: 0, path: '/k2.png' }))
    await waitFor(() => expect(invalidate).toHaveBeenCalledWith({ queryKey: ['job', 'j1'] }))

    act(() => es.emit('end', { id: 5, job_id: 'j1' }))   // the live segment really finished
    expect(es.closed).toBe(true)
    expect(result.current.ended).toBe(true)
    expect(result.current.events.some((e) => e.type === 'open')).toBe(false)  // a marker, not history
  })

  it('still stops on a final end when nothing is running', () => {
    renderHook(() => useJobEvents('j1'), { wrapper })
    const es = FakeEventSource.instances[0]
    act(() => es.emit('open', { job_id: 'j1', last_id: 9, active: false, status: 'done' }))
    act(() => es.emit('end', { id: 9, job_id: 'j1', final: true }))
    expect(es.closed).toBe(true)
  })
})
