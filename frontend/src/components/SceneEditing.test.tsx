import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { GatePanel } from './GatePanel'
import type { AssetOut, JobPublic } from '@/api/types'

const scene = (index: number, extra: Record<string, unknown> = {}) => ({
  index, description: 'd', character: '', setting: '', mood: '', continuous: false, status: 'complete',
  keyframe_prompt: 'kf', animation_prompt: 'an', keyframe_negative: '', animation_negative: '', mode: 't2i', references: [],
  keyframe_path: '/kf.png', keyframe_url: '/media/kf.png', keyframe_verdict: null, video_path: '/v.mp4', video_url: '/media/v.mp4',
  video_verdict: null, swapped_path: null, swapped_url: null, swap_verdict: null, variants: [], raw_video_path: '/v.mp4', post_paths: [],
  identity: '', identity_asset_id: null, identity_url: null, reference_asset_ids: [], keyframe_source: 'generated', keyframe_asset_id: null, clips: [],
  ...extra,
})
const job = (over: Record<string, unknown> = {}) => ({
  id: 'j1', idea: 'x', kind: 'video', face_swap: false, project_id: 'p1', status: 'done', active: false, stopping: false,
  edit_reference_slots: 2, pending_gate: null, scenes: [scene(0)], ...over,
}) as unknown as JobPublic
const asset = (id: string, label: string): AssetOut => ({
  id, kind: 'reference', name: `${id}.jpg`, label, label_source: 'human', note_caption: '', note_tags: [], note_text: '', describe_state: '', describe_error: '', thumb_url: `/api/projects/p1/assets/${id}/thumb`, sha256: '', created_at: '',
  path: `/a/${id}.jpg`, url: `/api/projects/p1/assets/${id}/file`,
})
const assets = [asset('tower', 'striped tower'), asset('bela', 'Bela portrait'), asset('choc', 'chocolate')]

type Call = { method: string; path: string; body: unknown }
function stub(calls: Call[]) {
  vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const raw = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url
    const path = new URL(raw, 'http://localhost').pathname
    const method = init?.method ?? (input instanceof Request ? input.method : 'GET')
    let body: unknown = null
    if (init?.body instanceof FormData) body = init.body
    else if (init?.body) body = JSON.parse(String(init.body))
    else if (input instanceof Request && method !== 'GET') body = await input.json()
    calls.push({ method, path, body })
    const payload = path.endsWith('/assets') ? asset('uploaded', 'new photo') : job()
    return new Response(JSON.stringify(payload), { status: 200, headers: { 'content-type': 'application/json' } })
  }))
}

const renderPanel = (j: JobPublic) => {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(<QueryClientProvider client={qc}><GatePanel job={j} assets={assets} /></QueryClientProvider>)
}

describe('editing scenes at any stage', () => {
  afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks() })

  it('is available on a stopped job and on an orphaned one, not on a live run', () => {
    const { unmount } = render(<QueryClientProvider client={new QueryClient()}><GatePanel job={job({ status: 'cancelled' })} assets={assets} /></QueryClientProvider>)
    expect(screen.getByText('Save changes')).toBeInTheDocument()
    unmount()
    const live = render(<QueryClientProvider client={new QueryClient()}><GatePanel job={job({ status: 'running', active: true })} assets={assets} /></QueryClientProvider>)
    expect(screen.queryByText('Save changes')).toBeNull()
    live.unmount()
    render(<QueryClientProvider client={new QueryClient()}><GatePanel job={job({ status: 'running', active: false })} assets={assets} /></QueryClientProvider>)
    expect(screen.getByText('Save changes')).toBeInTheDocument()
  })

  it('offers mode and references on a scene that already has a keyframe, applied on Regenerate only', async () => {
    const calls: Call[] = []
    stub(calls)
    vi.spyOn(window, 'confirm').mockReturnValue(true)
    const user = userEvent.setup()
    renderPanel(job())
    expect(screen.getByText(/apply on Regenerate keyframe/)).toBeInTheDocument()
    await user.selectOptions(screen.getByRole('combobox'), 'edit')
    await user.click(screen.getByTestId('refs-0-choose'))
    await user.click(screen.getByTestId('refs-0-tower'))
    await user.click(screen.getByTestId('refs-0-bela'))

    await user.click(screen.getByText('Save changes'))
    await waitFor(() => expect(calls.some((c) => c.method === 'PUT')).toBe(true))
    const saved = calls.find((c) => c.method === 'PUT')!.body as { scenes: { mode: string | null; reference_asset_ids: string[] | null }[] }
    expect(saved.scenes[0].mode).toBeNull()                   // no half-applied state on a made scene
    expect(saved.scenes[0].reference_asset_ids).toBeNull()

    await user.click(screen.getByText('Regenerate keyframe'))
    await waitFor(() => expect(calls.some((c) => c.path.endsWith('/regenerate'))).toBe(true))
    const regen = calls.find((c) => c.path.endsWith('/regenerate'))!.body as { mode: string; reference_asset_ids: string[] }
    expect(regen.mode).toBe('edit')
    expect(regen.reference_asset_ids).toEqual(['tower', 'bela'])  // click order = priority
  })

  it('warns when more references are picked than the edit workflow uses', async () => {
    const user = userEvent.setup()
    renderPanel(job({ scenes: [scene(0, { keyframe_path: null, mode: 'edit' })] }))
    expect(screen.getByText(/the first 2 are used/)).toBeInTheDocument()
    await user.click(screen.getByTestId('refs-0-choose'))
    for (const id of ['tower', 'bela', 'choc']) await user.click(screen.getByTestId(`refs-0-${id}`))
    expect(screen.getByText(/Only the first 2 are used/)).toBeInTheDocument()
  })

  it('uploads a reference from the card and adds it to the scene', async () => {
    const calls: Call[] = []
    stub(calls)
    const user = userEvent.setup()
    renderPanel(job({ scenes: [scene(0, { keyframe_path: null, mode: 'edit' })] }))
    const file = new File(['img'], 'harbour.jpg', { type: 'image/jpeg' })
    await user.click(screen.getByTestId('refs-0-choose'))
    await user.upload(screen.getByTestId('refs-0-upload'), file)
    await waitFor(() => expect(calls.some((c) => c.path === '/api/projects/p1/assets')).toBe(true))
    const form = calls.find((c) => c.path === '/api/projects/p1/assets')!.body as FormData
    expect(form.get('kind')).toBe('reference')
    await user.click(screen.getByText('Save changes'))
    await waitFor(() => expect(calls.some((c) => c.method === 'PUT')).toBe(true))
    const saved = calls.find((c) => c.method === 'PUT')!.body as { scenes: { reference_asset_ids: string[] }[] }
    expect(saved.scenes[0].reference_asset_ids).toEqual(['uploaded'])
  })

  it('uses a chosen project image as the keyframe', async () => {
    const calls: Call[] = []
    stub(calls)
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(true)
    const user = userEvent.setup()
    renderPanel(job())
    await user.click(screen.getByTestId('use-image-0'))
    await user.click(within(screen.getByTestId('keyframe-image-0')).getByTestId('keyframe-image-0-choc'))
    expect(confirm.mock.calls[0][0]).toMatch(/Use “chocolate” as the keyframe of scene 1\? Its face swap and clip are discarded/)
    await waitFor(() => expect(calls.some((c) => c.path === '/api/jobs/j1/scenes/0/regenerate')).toBe(true))
    const body = calls.find((c) => c.path.endsWith('/regenerate'))!.body as { what: string; keyframe_asset_id: string }
    expect(body).toMatchObject({ what: 'keyframe', keyframe_asset_id: 'choc' })
  })

  it('uses an uploaded image as the keyframe', async () => {
    const calls: Call[] = []
    stub(calls)
    vi.spyOn(window, 'confirm').mockReturnValue(true)
    const user = userEvent.setup()
    renderPanel(job())
    await user.click(screen.getByTestId('use-image-0'))
    await user.upload(screen.getByTestId('keyframe-image-0-upload'), new File(['x'], 'bela.jpg', { type: 'image/jpeg' }))
    await waitFor(() => expect(calls.some((c) => c.path.endsWith('/regenerate'))).toBe(true))
    expect(calls.find((c) => c.path.endsWith('/regenerate'))!.body).toMatchObject({ keyframe_asset_id: 'uploaded' })
  })

  it('has no image controls for a job without a project', () => {
    renderPanel(job({ project_id: null }))
    expect(screen.queryByTestId('use-image-0')).toBeNull()
  })
})
