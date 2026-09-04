import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { AssemblyVersions, describeCuts, describeSpeeds } from './AssemblyVersions'
import type { AssemblyVersion } from '@/api/types'

const versions: AssemblyVersion[] = [
  { version: 1, path: '/a1', url: '/media/a1.mp4?v=1', created_at: '', duration: 5.6, n_clips: 2, spec: { cuts: [{ transition: 'fade', duration: 0.5 }] }, size_bytes: 2048, fps: 16, width: 832, height: 480, exists: true, current: false },
  { version: 2, path: '/a2', url: '/media/a2.mp4?v=2', created_at: '', duration: 6.1, n_clips: 2, spec: { cuts: [{ transition: 'none', duration: 0.5 }] }, size_bytes: 4096, fps: 16, width: 832, height: 480, exists: true, current: true },
]

describe('AssemblyVersions', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('describes cuts', () => {
    expect(describeCuts({ cuts: [{ transition: 'fade', duration: 0.5 }, { transition: 'none', duration: 1 }] })).toBe('fade 0.5s · cut')
    expect(describeCuts({ cuts: [] })).toBe('single clip')
    expect(describeSpeeds({ items: [{ scene_index: 0, target_seconds: 6.125, speed: 0.5 }, { scene_index: 1 }], smooth: true })).toBe('s01 0.5× · smooth')
    expect(describeSpeeds({ items: [{ scene_index: 1 }] })).toBe('')
  })

  it('lists newest first, marks the current one, selects and deletes through the API', async () => {
    const calls: { method: string; path: string }[] = []
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const raw = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url
      const method = init?.method ?? (input instanceof Request ? input.method : 'GET')
      calls.push({ method, path: new URL(raw, 'http://x').pathname })
      return new Response(JSON.stringify({ available: true, active: false, transitions: [], default_spec: {}, clips: [], versions: [] }), { status: 200, headers: { 'content-type': 'application/json' } })
    }))
    vi.stubGlobal('confirm', () => true)
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(<QueryClientProvider client={qc}><AssemblyVersions jobId="j1" versions={versions} size={[832, 480]} /></QueryClientProvider>)
    const rows = screen.getAllByTestId(/version-/)
    expect(rows.map((r) => r.getAttribute('data-testid'))).toEqual(['version-2', 'version-1'])
    expect(screen.getByText('current · exported')).toBeInTheDocument()
    expect(screen.getByText('fade 0.5s')).toBeInTheDocument()

    const user = userEvent.setup()
    await user.click(screen.getByText('Use for export'))
    await waitFor(() => expect(calls.some((c) => c.method === 'POST' && c.path === '/api/jobs/j1/assemble/versions/1/select')).toBe(true))
    await user.click(screen.getAllByText('Delete')[0])
    await waitFor(() => expect(calls.some((c) => c.method === 'DELETE' && c.path === '/api/jobs/j1/assemble/versions/2')).toBe(true))
  })
})
