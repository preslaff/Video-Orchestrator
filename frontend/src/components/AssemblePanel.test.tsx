import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { AssemblePanel } from './AssemblePanel'
import type { JobPublic } from '@/api/types'

const job = {
  id: 'j1', idea: 'x', style: '', master_face: '', autonomy: 'semi', n_scenes: 3, seed: 0, max_attempts: 2, evaluate: true,
  face_swap: false, mode: 't2i', reference_images: [], project_id: null, profile_id: null, kind: 'video', export: null, assembly: null,
  profile: { video: { width: 832, height: 480, fps: 16, length: 49 } }, status: 'done', created_at: '', updated_at: '', pending_gate: null,
  scenes: [0, 1, 2].map((i) => ({ index: i, description: '', character: '', setting: '', mood: '', continuous: false, status: 'complete', keyframe_prompt: '', animation_prompt: '', keyframe_negative: '', animation_negative: '', mode: 't2i', references: [], keyframe_path: null, keyframe_url: null, keyframe_verdict: null, video_path: `/v${i}.mp4`, video_url: `/media/v${i}.mp4`, video_verdict: null, swapped_path: null, swapped_url: null, swap_verdict: null, variants: [], raw_video_path: null, post_paths: [] })),
} as unknown as JobPublic

const STATUS = {
  available: true, active: false, transitions: ['none', 'fade', 'wipeleft', 'slideup'],
  default_spec: { items: [0, 1, 2].map((i) => ({ scene_index: i, source: 'final' })), cuts: [{ transition: 'fade', duration: 0.5 }, { transition: 'fade', duration: 0.5 }], fps: null, width: null, height: null, crf: 18, encoder: 'libx264' },
  clips: [0, 1, 2].map((i) => ({ scene_index: i, source: 'final', path: `/v${i}.mp4`, url: `/media/v${i}.mp4`, duration: 3.062, fps: 16, width: 832, height: 480, has_raw: false })),
  profile_fps: 16, record: null,
  deliveries: [
    { id: 'source', label: 'Source (as rendered)', width: null, height: null, fps: null, note: '832×480 @ 16 fps' },
    { id: 'master_1080', label: 'Master 1080p', width: 1920, height: 1080, fps: 24, note: 'for an edit or a screening' },
    { id: 'social_1080', label: 'Social 1080', width: 1920, height: 1080, fps: 30, note: 'Reels / TikTok / Shorts' },
  ],
  delivery: 'master_1080',
}

function mockFetch(status: Record<string, unknown> = STATUS) {
  const posts: unknown[] = []
  vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const raw = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url
    const method = init?.method ?? (input instanceof Request ? input.method : 'GET')
    const json = (d: unknown, status = 200) => new Response(JSON.stringify(d), { status, headers: { 'content-type': 'application/json' } })
    if (new URL(raw, 'http://x').pathname === '/api/jobs/j1/assemble') {
      if (method === 'POST') {
        posts.push(init?.body ? JSON.parse(String(init.body)) : await (input as Request).json())
        return json({ ...status, active: true })
      }
      return json(status)
    }
    return json({ error: { code: 'not_found', message: raw } }, 404)
  }))
  return posts
}

function renderPanel(onStarted = vi.fn()) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(<QueryClientProvider client={qc}><AssemblePanel job={job} events={[]} onStarted={onStarted} /></QueryClientProvider>)
  return onStarted
}

describe('AssemblePanel', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('renders clips, updates the total when a clip is unticked and a cut becomes none, and posts the spec in order', async () => {
    const posts = mockFetch()
    const onStarted = renderPanel()
    const user = userEvent.setup()
    await screen.findByText('Scene 1')
    expect(screen.getByTestId('total')).toHaveTextContent('3 clip(s)')
    expect(screen.getByTestId('total')).toHaveTextContent('8.19s')          // 3×3.062 − 2×0.5
    expect(screen.getAllByLabelText('transition')).toHaveLength(2)

    await user.click(screen.getByLabelText('include scene 3'))
    expect(screen.getByTestId('total')).toHaveTextContent('2 clip(s)')
    expect(screen.getAllByLabelText('transition')).toHaveLength(1)
    expect(screen.getByTestId('total')).toHaveTextContent('5.62s')

    await user.selectOptions(screen.getByLabelText('transition'), 'none')
    expect(screen.queryByLabelText('transition seconds')).toBeNull()
    expect(screen.getByTestId('total')).toHaveTextContent('6.12s')

    await user.click(screen.getAllByLabelText('move down')[0])              // scene 1 below scene 2
    await user.click(screen.getByTestId('assemble'))
    await waitFor(() => expect(posts).toHaveLength(1))
    expect(posts[0]).toMatchObject({ items: [{ scene_index: 1, source: 'final', target_seconds: null }, { scene_index: 0, source: 'final', target_seconds: null }], cuts: [{ transition: 'none' }], encoder: 'libx264', fps: null, smooth: false })
    await waitFor(() => expect(onStarted).toHaveBeenCalled())
    expect(within(screen.getByTestId('assemble')).getByText('Assembling…')).toBeInTheDocument()
  })
})


describe('AssemblePanel speed', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('a target duration retimes the clip, updates the total and the chip, and is posted with the smooth flag', async () => {
    const posts = mockFetch()
    renderPanel()
    const user = userEvent.setup()
    await screen.findByText('Scene 1')
    const field = screen.getByLabelText('scene 1 seconds')
    await user.clear(field)
    await user.type(field, '6.124')
    expect(screen.getByTestId('speed-0')).toHaveTextContent('0.5×')
    expect(screen.getByTestId('total')).toHaveTextContent('11.24s')   // 6.12 (clamped/rounded) + 2×3.062 − 2×0.5
    await user.click(screen.getByLabelText('smooth slow motion'))
    await user.click(screen.getByTestId('assemble'))
    await waitFor(() => expect(posts).toHaveLength(1))
    expect(posts[0]).toMatchObject({ items: [{ scene_index: 0, target_seconds: 6.12 }, { scene_index: 1, target_seconds: null }, { scene_index: 2, target_seconds: null }], smooth: true })
    await user.click(screen.getByText('natural'))
    expect(screen.getByTestId('total')).toHaveTextContent('8.19s')
    expect(screen.getByTestId('speed-0')).toHaveTextContent('1×')
  })

  it('says so when the clips changed since the assembly', async () => {
    const record = { path: '/out/assembly_v1.mp4', url: '/media/assembly_v1.mp4', created_at: '', duration: 8.2, n_clips: 3, spec: {},
      ok: true, error: '', size_bytes: 10, fps: 16, width: 832, height: 480, exists: true, exported: false, version: 1, clips: [], stale: true }
    mockFetch({ ...STATUS, record })
    renderPanel()
    expect(await screen.findByTestId('assembly-stale')).toHaveTextContent('Clips changed since this assembly')
  })

  it('masters to a delivery target and says when that is an upscale', async () => {
    const posts = mockFetch()
    renderPanel()
    const user = userEvent.setup()
    await screen.findByText('Scene 1')
    expect(screen.getByLabelText('delivery')).toHaveValue('master_1080')   // the job's default
    expect(screen.getByTestId('delivery-format')).toHaveTextContent('1920×1080 @ 24 fps')
    expect(screen.getByTestId('delivery-upscale')).toHaveTextContent('upscaled 2.3× from 832×480')

    await user.selectOptions(screen.getByLabelText('grain'), '1')
    await user.click(screen.getByTestId('assemble'))
    await waitFor(() => expect(posts).toHaveLength(1))
    expect(posts[0]).toMatchObject({ delivery: 'master_1080', grain: 1, fps: null })
  })

  it('keeps the clips\' own format when Source is chosen', async () => {
    const posts = mockFetch()
    renderPanel()
    const user = userEvent.setup()
    await screen.findByText('Scene 1')
    await user.selectOptions(screen.getByLabelText('delivery'), 'source')
    expect(screen.queryByTestId('delivery-upscale')).toBeNull()
    await user.click(screen.getByTestId('assemble'))
    await waitFor(() => expect(posts).toHaveLength(1))
    expect(posts[0]).toMatchObject({ delivery: 'source' })
  })
})

