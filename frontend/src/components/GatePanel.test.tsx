import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { GatePanel } from './GatePanel'
import type { AssetOut, JobPublic } from '@/api/types'

const scene = (index: number, extra: Partial<JobPublic['scenes'][number]> = {}) => ({
  index, description: 'd', character: '', setting: '', mood: '', continuous: false, status: 'pending',
  keyframe_prompt: 'kf', animation_prompt: 'an', keyframe_negative: '', animation_negative: '', mode: 't2i', references: [],
  keyframe_path: null, keyframe_url: null, keyframe_verdict: null, video_path: null, video_url: null, video_verdict: null,
  swapped_path: null, swapped_url: null, swap_verdict: null, variants: [], raw_video_path: null, post_paths: [],
  identity: '', identity_asset_id: null, identity_url: null, reference_asset_ids: [], clips: [], ...extra,
})
const job = {
  id: 'j1', idea: 'ages', style: '', master_face: '/m.png', autonomy: 'semi', n_scenes: 2, seed: 0, max_attempts: 2,
  evaluate: true, face_swap: true, mode: 't2i', reference_images: [], project_id: 'p1', profile_id: null, profile: null,
  kind: 'video', export: null, status: 'needs_approval', pending_gate: 'storyboard', created_at: '', updated_at: '',
  scenes: [scene(0, { identity: '/faces/young.png', identity_asset_id: 'young' }), scene(1)],
} as unknown as JobPublic
const asset = (id: string, kind: string, label: string, extra: Partial<AssetOut> = {}): AssetOut => ({ id, kind, name: `${id}.png`, label, label_source: 'human', note_caption: '', note_tags: [], note_text: '', describe_state: '', describe_error: '', thumb_url: `/api/projects/p1/assets/${id}/thumb`, sha256: '', created_at: '', path: `/faces/${id}.png`, url: `/api/projects/p1/assets/${id}/file`, ...extra })
const assets = [asset('young', 'master_face', 'Maria, age 8'), asset('old', 'master_face', 'Maria, age 70'), asset('room', 'reference', 'the room')]

/** fetch stub: records PUT /api/jobs/j1/scenes bodies (openapi-fetch passes a Request object). */
function stubFetch(puts: unknown[]) {
  vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const raw = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url
    const url = new URL(raw, 'http://localhost').pathname
    const method = init?.method ?? (input instanceof Request ? input.method : 'GET')
    const json = (data: unknown, status = 200) => new Response(JSON.stringify(data), { status, headers: { 'content-type': 'application/json' } })
    if (url === '/api/jobs/j1/scenes' && method === 'PUT') {
      puts.push(init?.body ? JSON.parse(String(init.body)) : input instanceof Request ? await input.json() : null)
      return json(job)
    }
    return json({ error: { code: 'not_found', message: url } }, 404)
  }))
}

describe('GatePanel identity picker', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('sends the chosen face per scene (and "" for none) when saving', async () => {
    const puts: unknown[] = []
    stubFetch(puts)
    const user = userEvent.setup()
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(<QueryClientProvider client={qc}><GatePanel job={job} assets={assets} /></QueryClientProvider>)
    expect(screen.queryByTestId('identity-0-young')).toBeNull()        // pickers start collapsed
    expect(screen.getAllByText('Maria, age 8').length).toBeGreaterThan(0) // ...showing what is chosen
    await user.click(screen.getByTestId('identity-0-choose'))
    expect(screen.getByTestId('identity-0-young')).toHaveAttribute('aria-pressed', 'true')
    await user.click(screen.getByTestId('identity-1-choose'))         // opening another closes the first
    expect(screen.queryByTestId('identity-0-young')).toBeNull()
    await user.click(screen.getByTestId('identity-1-old'))            // scene 2 gets the older face
    await user.click(screen.getByTestId('identity-0-choose'))
    await user.click(screen.getByTestId('identity-0-young'))          // toggling the chosen face off = none
    await user.click(screen.getByText('Save changes'))
    await waitFor(() => expect(puts).toHaveLength(1))
    const body = puts[0] as { scenes: { index: number; identity_asset_id: string | null; reference_asset_ids: string[] | null }[] }
    expect(body.scenes[0].identity_asset_id).toBe('')
    expect(body.scenes[1].identity_asset_id).toBe('old')
    expect(body.scenes[0].reference_asset_ids).toEqual([])            // t2i scene: no refs picked
  })

  it('leaves identity untouched when the project has no faces', async () => {
    const puts: unknown[] = []
    stubFetch(puts)
    const user = userEvent.setup()
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(<QueryClientProvider client={qc}><GatePanel job={job} assets={[]} /></QueryClientProvider>)
    expect(screen.queryByTestId('identity-0')).toBeNull()
    await user.click(screen.getByText('Save changes'))
    await waitFor(() => expect(puts).toHaveLength(1))
    expect((puts[0] as { scenes: { identity_asset_id: unknown }[] }).scenes[0].identity_asset_id).toBeNull()
  })
})

describe('GatePanel reference hints', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('shows the storyboard hint and applies it only when accepted', async () => {
    const puts: unknown[] = []
    stubFetch(puts)
    const user = userEvent.setup()
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const hinted = {
      ...job, face_swap: false,
      scenes: [
        scene(0, { reference_suggestion: { asset_id: 'room', why: 'same room from the door', fit: 'same_shot' } }),
        scene(1, { reference_suggestion: { asset_id: 'gone', why: 'deleted photo', fit: 'same_shot' } }),
      ],
    } as unknown as JobPublic
    render(<QueryClientProvider client={qc}><GatePanel job={hinted} assets={assets} /></QueryClientProvider>)
    expect(screen.getByText('same room from the door')).toBeInTheDocument()
    expect(screen.queryByTestId('hint-1')).toBeNull()                    // unknown asset: no hint
    await user.click(screen.getByText('Save changes'))                  // not accepted: scene stays t2i
    await waitFor(() => expect(puts).toHaveLength(1))
    type Body = { scenes: { mode: string | null; reference_asset_ids: string[] | null }[] }
    expect((puts[0] as Body).scenes[0]).toMatchObject({ mode: 't2i', reference_asset_ids: [] })
    await user.click(screen.getByTestId('hint-0-use'))
    expect(screen.getByTestId('hint-0-undo')).toBeInTheDocument()
    await user.click(screen.getByText('Save changes'))
    await waitFor(() => expect(puts).toHaveLength(2))
    expect((puts[1] as Body).scenes[0]).toMatchObject({ mode: 'edit', reference_asset_ids: ['room'] })
    await user.click(screen.getByTestId('hint-0-undo'))
    expect(screen.getByTestId('hint-0-use')).toBeInTheDocument()
  })
})

describe('GatePanel save keyframe as project reference', () => {
  afterEach(() => vi.unstubAllGlobals())

  /** fetch stub that records POSTs to the save-asset route and answers with an asset. */
  function stubSave(posts: { url: string; body: unknown }[], status = 201) {
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const raw = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url
      const url = new URL(raw, 'http://localhost').pathname
      const body = init?.body ? JSON.parse(String(init.body)) : input instanceof Request ? await input.json() : null
      posts.push({ url, body })
      return new Response(JSON.stringify(asset('new', 'reference', 'scene-1-keyframe.png')),
        { status, headers: { 'content-type': 'application/json' } })
    }))
  }

  const withKeyframe = {
    ...job, face_swap: false,
    scenes: [scene(0, { keyframe_path: '/k0.png', keyframe_url: '/media/k0.png' }), scene(1)],
  } as unknown as JobPublic

  it('copies the keyframe into the project and says so', async () => {
    const posts: { url: string; body: unknown }[] = []
    stubSave(posts)
    const user = userEvent.setup()
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(<QueryClientProvider client={qc}><GatePanel job={withKeyframe} assets={assets} /></QueryClientProvider>)
    expect(screen.queryByTestId('save-asset-1')).toBeNull()        // scene 2 has no keyframe yet
    await user.click(screen.getByTestId('save-asset-0'))
    await waitFor(() => expect(posts).toHaveLength(1))
    expect(posts[0].url).toBe('/api/jobs/j1/scenes/0/save-asset')
    expect(posts[0].body).toMatchObject({ kind: 'reference' })
    expect(await screen.findByTestId('save-asset-done-0')).toHaveTextContent('Saved to the project')
  })

  it('says when the project already holds that frame, and hides the button without a project', async () => {
    const posts: { url: string; body: unknown }[] = []
    stubSave(posts, 200)
    const user = userEvent.setup()
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const { unmount } = render(<QueryClientProvider client={qc}><GatePanel job={withKeyframe} assets={assets} /></QueryClientProvider>)
    await user.click(screen.getByTestId('save-asset-0'))
    expect(await screen.findByTestId('save-asset-done-0')).toHaveTextContent('Already in the project')
    unmount()

    const free = { ...withKeyframe, project_id: null } as unknown as JobPublic
    render(<QueryClientProvider client={new QueryClient()}><GatePanel job={free} assets={[]} /></QueryClientProvider>)
    expect(screen.queryByTestId('save-asset-0')).toBeNull()
  })
})
