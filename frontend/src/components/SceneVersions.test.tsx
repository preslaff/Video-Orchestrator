import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { GatePanel } from './GatePanel'
import type { JobPublic } from '@/api/types'

const variant = (id: string, path: string, extra: Record<string, unknown> = {}) => ({
  id, path, url: `/media/${id}.png`, seed: 7, verdict: null, swapped_path: null, swapped_url: null,
  post_paths: [], source: 'generated', asset_id: null, created_at: '', batch: 0, active: false, ...extra,
})
const clip = (id: string, extra: Record<string, unknown> = {}) => ({
  id, video_path: `/${id}.mp4`, url: `/media/${id}.mp4`, raw_video_path: null, raw_url: null, post_paths: [],
  seed: 3, verdict: null, keyframe_path: '/kf1.png', created_at: '', batch: 0, active: false, ...extra,
})
const scene = (extra: Record<string, unknown> = {}) => ({
  index: 0, description: 'd', character: '', setting: '', mood: '', continuous: false, status: 'complete',
  keyframe_prompt: 'kf', animation_prompt: 'an', keyframe_negative: '', animation_negative: '', mode: 't2i',
  references: [], keyframe_path: '/kf2.png', keyframe_url: '/media/kf2.png', keyframe_verdict: null,
  video_path: '/c2.mp4', video_url: '/media/c2.mp4', video_verdict: null, swapped_path: null, swapped_url: null,
  swap_verdict: null, raw_video_path: null, post_paths: [], identity: '', identity_asset_id: null,
  identity_url: null, reference_asset_ids: [], keyframe_source: 'generated', keyframe_asset_id: null,
  variants: [variant('k1', '/kf1.png'), variant('k2', '/kf2.png', { active: true, batch: 1 })],
  clips: [clip('c1'), clip('c2', { active: true, batch: 1 })],
  ...extra,
})
const job = (over: Record<string, unknown> = {}) => ({
  id: 'j1', idea: 'x', kind: 'video', face_swap: false, project_id: null, status: 'done', active: false,
  stopping: false, edit_reference_slots: 2, pending_gate: null, profile: null, scenes: [scene()], ...over,
}) as unknown as JobPublic

type Call = { method: string; path: string; body: unknown }
function stub(calls: Call[], reply: () => unknown = () => job()) {
  vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const raw = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url
    const method = init?.method ?? (input instanceof Request ? input.method : 'GET')
    let body: unknown = null
    if (init?.body) body = JSON.parse(String(init.body))
    else if (input instanceof Request && method !== 'GET') body = await input.json().catch(() => null)  // DELETE has none
    calls.push({ method, path: new URL(raw, 'http://localhost').pathname, body })
    return new Response(JSON.stringify(reply()), { status: 200, headers: { 'content-type': 'application/json' } })
  }))
}
const renderPanel = (j: JobPublic) =>
  render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}><GatePanel job={j} /></QueryClientProvider>)

describe('scene media and its versions', () => {
  afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks() })

  it('shows the keyframe and the clip next to the editor', () => {
    renderPanel(job())
    expect(screen.getByAltText('scene 1 keyframe')).toHaveAttribute('src', '/media/kf2.png')
    const clips = document.querySelectorAll('video')
    expect(clips).toHaveLength(1)
    expect(clips[0]).toHaveAttribute('src', '/media/c2.mp4')
    expect(screen.getByDisplayValue('kf')).toBeInTheDocument()  // the prompt editor is in the same row
  })

  it('marks the active version and selects another one straight away', async () => {
    const calls: Call[] = []
    stub(calls)
    const strip = () => screen.getByTestId('versions-keyframe-0')
    renderPanel(job())
    expect(within(strip()).getByTestId('version-keyframe-0-k2')).toHaveAttribute('aria-pressed', 'true')
    await userEvent.click(within(strip()).getByTestId('version-keyframe-0-k1'))
    await waitFor(() => expect(calls.some((c) => c.method === 'PUT')).toBe(true))
    expect(calls.find((c) => c.method === 'PUT')!.body).toEqual({ scenes: [{ index: 0, chosen_variant_id: 'k1' }] })
  })

  it('selects a clip version, which is what assembly will use', async () => {
    const calls: Call[] = []
    stub(calls)
    renderPanel(job())
    await userEvent.click(screen.getByTestId('version-clip-0-c1'))
    await waitFor(() => expect(calls.some((c) => c.method === 'PUT')).toBe(true))
    expect(calls.find((c) => c.method === 'PUT')!.body).toEqual({ scenes: [{ index: 0, chosen_clip_id: 'c1' }] })
  })

  it('deletes a version after saying the file is kept', async () => {
    const calls: Call[] = []
    stub(calls)
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(true)
    renderPanel(job())
    await userEvent.click(screen.getByTestId('version-delete-keyframe-0-k1'))
    expect(confirm.mock.calls[0][0]).toMatch(/The file stays on disk/)
    await waitFor(() => expect(calls.some((c) => c.method === 'DELETE')).toBe(true))
    expect(calls.find((c) => c.method === 'DELETE')!.path).toBe('/api/jobs/j1/scenes/0/versions/k1')
  })

  it('hides a strip when there is nothing to choose between', () => {
    renderPanel(job({ scenes: [scene({ variants: [variant('k1', '/kf1.png', { active: true })], clips: [] })] }))
    expect(screen.queryByTestId('versions-keyframe-0')).toBeNull()
    expect(screen.queryByTestId('versions-clip-0')).toBeNull()
  })

  it('keeps a half-typed prompt when the job refetches with new media', async () => {
    const user = userEvent.setup()
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const { rerender } = render(
      <QueryClientProvider client={qc}><GatePanel job={job()} /></QueryClientProvider>)
    const box = screen.getByDisplayValue('kf')
    await user.clear(box)
    await user.type(box, 'a lighthouse at dusk')

    // the server answers with a regenerated keyframe and an unrelated prompt change
    const updated = job({ scenes: [scene({ keyframe_url: '/media/kf3.png', description: 'from the server' })] })
    rerender(<QueryClientProvider client={qc}><GatePanel job={updated} /></QueryClientProvider>)

    expect(screen.getByDisplayValue('a lighthouse at dusk')).toBeInTheDocument()   // my edit survives
    expect(screen.getByDisplayValue('from the server')).toBeInTheDocument()        // the clean field follows
    expect(screen.getByAltText('scene 1 keyframe')).toHaveAttribute('src', '/media/kf3.png')
  })
})
