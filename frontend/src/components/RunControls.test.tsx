import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { RunControls } from './RunControls'
import type { JobPublic } from '@/api/types'

const job = (over: Partial<JobPublic>) => ({
  id: 'j1', idea: 'x', status: 'running', active: true, stopping: false, scenes: [], ...over,
}) as unknown as JobPublic

function stub(calls: string[]) {
  vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const raw = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url
    const method = init?.method ?? (input instanceof Request ? input.method : 'GET')
    calls.push(`${method} ${new URL(raw, 'http://localhost').pathname}`)
    return new Response(JSON.stringify(job({})), { status: 200, headers: { 'content-type': 'application/json' } })
  }))
}

const renderWith = (j: JobPublic, onStarted = vi.fn()) => {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(<QueryClientProvider client={qc}><RunControls job={j} onStarted={onStarted} /></QueryClientProvider>)
  return onStarted
}

describe('RunControls', () => {
  afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks() })

  it('stops a live run', async () => {
    const calls: string[] = []
    stub(calls)
    renderWith(job({}))
    expect(screen.queryByTestId('resume-job')).toBeNull()
    await userEvent.click(screen.getByTestId('stop-job'))
    await waitFor(() => expect(calls).toContain('POST /api/jobs/j1/stop'))
  })

  it('shows a pending stop as Stopping and does not offer it twice', () => {
    renderWith(job({ stopping: true }))
    expect(screen.getByTestId('stop-job')).toBeDisabled()
    expect(screen.getByText('Stopping…')).toBeInTheDocument()
  })

  it('resumes a stopped job and reconnects the stream', async () => {
    const calls: string[] = []
    stub(calls)
    const onStarted = renderWith(job({ status: 'cancelled', active: false }))
    expect(screen.queryByTestId('stop-job')).toBeNull()
    await userEvent.click(screen.getByTestId('resume-job'))
    await waitFor(() => expect(calls).toContain('POST /api/jobs/j1/retry'))
    await waitFor(() => expect(onStarted).toHaveBeenCalled())
  })

  it('treats a running job no server owns as orphaned: Resume after a confirm, no Stop', async () => {
    const calls: string[] = []
    stub(calls)
    const confirm = vi.spyOn(window, 'confirm').mockReturnValueOnce(false).mockReturnValueOnce(true)
    renderWith(job({ status: 'running', active: false }))
    expect(screen.queryByTestId('stop-job')).toBeNull()
    await userEvent.click(screen.getByTestId('resume-job'))
    expect(calls).toEqual([])  // declined
    await userEvent.click(screen.getByTestId('resume-job'))
    await waitFor(() => expect(calls).toContain('POST /api/jobs/j1/retry'))
    expect(confirm).toHaveBeenCalledTimes(2)
  })
})
