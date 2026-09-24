import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { GOTO_SCENE, GatePanel } from './GatePanel'
import type { AssetOut, JobPublic } from '@/api/types'

const scene = (index: number, extra: Partial<JobPublic['scenes'][number]> = {}) => ({
  index, id: `s${index}`, writing_prompts: false, description: 'd', character: '', setting: '', mood: '', continuous: false, status: 'pending',
  keyframe_prompt: 'kf', animation_prompt: 'an', keyframe_negative: '', animation_negative: '', mode: 't2i', edit_workflow: '', edit_base: '', edit_base_path: '', references: [],
  keyframe_path: null, keyframe_url: null, keyframe_verdict: null, video_path: null, video_url: null, video_verdict: null,
  swapped_path: null, swapped_url: null, swap_verdict: null, variants: [], raw_video_path: null, post_paths: [],
  identity: '', identity_asset_id: null, identity_url: null, reference_asset_ids: [], clips: [], locked_regions: [], narration: '', narration_url: null, narration_seconds: 0, narration_stale: false, ...extra,
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

describe('GatePanel locked regions', () => {
  afterEach(() => vi.unstubAllGlobals())

  const withClip = {
    ...job, face_swap: false,
    scenes: [
      scene(0, { keyframe_path: '/k0.png', keyframe_url: '/media/k0.png', video_path: '/v0.mp4',
                 video_url: '/media/v0.mp4' }),
      scene(1),
    ],
  } as unknown as JobPublic

  it('marks a label on the keyframe and sends it as fractions of the frame', async () => {
    const puts: { url: string; body: unknown }[] = []
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const raw = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url
      const body = init?.body ? JSON.parse(String(init.body)) : input instanceof Request ? await input.json() : null
      puts.push({ url: new URL(raw, 'http://localhost').pathname, body })
      return new Response(JSON.stringify(withClip), { status: 200, headers: { 'content-type': 'application/json' } })
    }))
    const user = userEvent.setup()
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(<QueryClientProvider client={qc}><GatePanel job={withClip} assets={assets} /></QueryClientProvider>)
    expect(screen.queryByTestId('lock-regions-1')).toBeNull()      // scene 2 has no frame to mark

    await user.click(screen.getByTestId('lock-regions-0'))
    const frame = screen.getByTestId('locks-0-frame')
    frame.getBoundingClientRect = () =>
      ({ left: 0, top: 0, width: 100, height: 100, right: 100, bottom: 100, x: 0, y: 0, toJSON: () => ({}) })
    fireEvent.pointerDown(frame, { clientX: 40, clientY: 30, pointerId: 1 })
    fireEvent.pointerMove(frame, { clientX: 60, clientY: 40, pointerId: 1 })
    fireEvent.pointerUp(frame, { clientX: 60, clientY: 40, pointerId: 1 })
    await user.click(screen.getByTestId('locks-0-save'))

    await waitFor(() => expect(puts).toHaveLength(1))
    expect(puts[0].url).toBe('/api/jobs/j1/scenes/0/locks')
    expect(puts[0].body).toEqual({ regions: [{ x: 0.4, y: 0.3, w: 0.2, h: 0.1 }] })
  })

  it('says how many regions a scene already has', () => {
    const marked = {
      ...withClip,
      scenes: [scene(0, { keyframe_url: '/media/k0.png', video_url: '/media/v0.mp4',
                          locked_regions: [{ x: 0.4, y: 0.3, w: 0.2, h: 0.1 }] })],
    } as unknown as JobPublic
    render(<QueryClientProvider client={new QueryClient()}><GatePanel job={marked} assets={assets} /></QueryClientProvider>)
    expect(screen.getByTestId('lock-regions-0')).toHaveTextContent('Locked: 1 region')
  })
})

describe('GatePanel save failure', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('names the restart when the running server lacks the route', async () => {
    vi.stubGlobal('fetch', vi.fn(async () =>
      new Response(JSON.stringify({ detail: 'Not Found' }), { status: 404, headers: { 'content-type': 'application/json' } })))
    const user = userEvent.setup()
    const withKeyframe = {
      ...job, face_swap: false, scenes: [scene(0, { keyframe_path: '/k0.png', keyframe_url: '/media/k0.png' })],
    } as unknown as JobPublic
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
    render(<QueryClientProvider client={qc}><GatePanel job={withKeyframe} assets={assets} /></QueryClientProvider>)
    await user.click(screen.getByTestId('save-asset-0'))
    expect(await screen.findByTestId('save-asset-error-0')).toHaveTextContent('restart vorch serve')
  })
})

describe('GatePanel with many scenes', () => {
  afterEach(() => vi.unstubAllGlobals())

  const bigJob = {
    ...job, face_swap: false,
    scenes: Array.from({ length: 8 }, (_, i) => scene(i, {
      description: `scene ${i + 1} description`,
      keyframe_url: i === 0 ? '/media/k0.png' : null,
      keyframe_path: i === 0 ? '/k0.png' : null,
    })),
  } as unknown as JobPublic

  const show = (j: JobPublic = bigJob) =>
    render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <GatePanel job={j} assets={assets} /></QueryClientProvider>)

  it('collapses the rows and expands the one you click', async () => {
    const user = userEvent.setup()
    show()
    expect(screen.getByTestId('scene-count')).toHaveTextContent('8 scenes · 0 open')
    expect(screen.queryByDisplayValue('kf')).toBeNull()                 // no editor rendered yet
    expect(screen.getByText('scene 4 description')).toBeInTheDocument() // but every row is listed

    await user.click(screen.getByTestId('scene-toggle-3'))
    expect(screen.getByTestId('scene-toggle-3')).toHaveAttribute('aria-expanded', 'true')
    expect(screen.getAllByDisplayValue('kf')).toHaveLength(1)           // only that scene's editor
    expect(screen.getByTestId('scene-count')).toHaveTextContent('1 open')

    await user.click(screen.getByTestId('toggle-all-scenes'))           // Collapse all
    expect(screen.queryByDisplayValue('kf')).toBeNull()
  })

  it('keeps an unsaved edit while collapsed and flags the row', async () => {
    const user = userEvent.setup()
    show()
    await user.click(screen.getByTestId('scene-toggle-2'))
    const prompt = screen.getAllByDisplayValue('kf')[0]
    await user.type(prompt, ' extra')
    expect(screen.getByTestId('scene-dirty-2')).toBeInTheDocument()
    await user.click(screen.getByTestId('scene-toggle-2'))              // collapse again
    expect(screen.queryByDisplayValue('kf extra')).toBeNull()
    expect(screen.getByTestId('scene-dirty-2')).toBeInTheDocument()     // the work is still there
    await user.click(screen.getByTestId('scene-toggle-2'))
    expect(screen.getByDisplayValue('kf extra')).toBeInTheDocument()
  })

  it('shows the keyframe thumbnail on a row that has one, and opens on a jump event', () => {
    show()
    const row = screen.getByTestId('scene-toggle-0')
    expect(row.querySelector('img')).toHaveAttribute('src', '/media/k0.png')
    expect(screen.getByTestId('scene-toggle-1').querySelector('img')).toBeNull()

    act(() => { window.dispatchEvent(new CustomEvent(GOTO_SCENE, { detail: 5 })) })
    expect(screen.getByTestId('scene-toggle-5')).toHaveAttribute('aria-expanded', 'true')
  })

  it('leaves a small job fully expanded', () => {
    show({ ...bigJob, scenes: bigJob.scenes.slice(0, 3) } as unknown as JobPublic)
    expect(screen.getAllByDisplayValue('kf')).toHaveLength(3)
    expect(screen.getByTestId('scene-count')).toHaveTextContent('3 scenes · 3 open')
  })
})

describe('GatePanel storyboard structure', () => {
  afterEach(() => vi.unstubAllGlobals())

  type Call = { method: string; path: string; body: unknown; query: string }

  /** Records every call and answers with the job the test hands it back. */
  function stubCalls(calls: Call[], reply: () => unknown) {
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const raw = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url
      const url = new URL(raw, 'http://localhost')
      const method = init?.method ?? (input instanceof Request ? input.method : 'GET')
      const body = init?.body ? JSON.parse(String(init.body)) : input instanceof Request ? await input.json().catch(() => null) : null
      // Only the mutations: the panel also GETs the registered edit workflows for its selector.
      if (method !== 'GET') calls.push({ method, path: url.pathname, body, query: url.search })
      return new Response(JSON.stringify(reply()), { status: 200, headers: { 'content-type': 'application/json' } })
    }))
  }

  const threeScenes = {
    ...job, face_swap: false,
    scenes: [scene(0, { description: 'first' }), scene(1, { description: 'second' }), scene(2, { description: 'third' })],
  } as unknown as JobPublic

  const show = (j: JobPublic = threeScenes) =>
    render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <GatePanel job={j} assets={assets} /></QueryClientProvider>)

  it('moves a scene with the arrows, sending the whole id order', async () => {
    const calls: Call[] = []
    stubCalls(calls, () => threeScenes)
    const user = userEvent.setup()
    show()
    expect(screen.getByTestId('scene-up-0')).toBeDisabled()          // nothing above the first
    expect(screen.getByTestId('scene-down-2')).toBeDisabled()
    await user.click(screen.getByTestId('scene-down-0'))
    await waitFor(() => expect(calls).toHaveLength(1))
    expect(calls[0].method).toBe('PUT')
    expect(calls[0].path).toBe('/api/jobs/j1/scenes/order')
    expect(calls[0].body).toEqual({ scene_ids: ['s1', 's0', 's2'] })
  })

  it('reorders by dropping a row into a gap', async () => {
    const calls: Call[] = []
    stubCalls(calls, () => threeScenes)
    show()
    // Drag scene 3 (id s2) into the gap before scene 1.
    const row = screen.getByTestId('scene-toggle-2').closest('[draggable]')!
    fireEvent.dragStart(row)
    fireEvent.dragOver(screen.getByTestId('scene-gap-0'))
    fireEvent.drop(screen.getByTestId('scene-gap-0'))
    await waitFor(() => expect(calls).toHaveLength(1))
    expect(calls[0].body).toEqual({ scene_ids: ['s2', 's0', 's1'] })
  })

  it('offers both delete choices and sends the one clicked', async () => {
    const calls: Call[] = []
    stubCalls(calls, () => threeScenes)
    const user = userEvent.setup()
    show()
    expect(screen.queryByTestId('confirm-delete-1')).toBeNull()
    await user.click(screen.getByTestId('scene-delete-1'))
    expect(screen.getByTestId('confirm-delete-1')).toHaveTextContent('Delete scene 2?')
    await user.click(screen.getByTestId('confirm-delete-files-1'))
    await waitFor(() => expect(calls).toHaveLength(1))
    expect(calls[0].method).toBe('DELETE')
    expect(calls[0].path).toBe('/api/jobs/j1/scenes/1')
    expect(calls[0].query).toContain('delete_files=true')
  })

  it('adds a scene from a description and shows the writing marker', async () => {
    const calls: Call[] = []
    stubCalls(calls, () => threeScenes)
    const user = userEvent.setup()
    show()
    await user.click(screen.getByTestId('add-scene-1'))               // the gap before scene 2
    await user.type(screen.getByTestId('add-scene-description'), 'the lamp at night')
    await user.type(screen.getByTestId('add-scene-setting'), 'Atlantic coast')
    await user.click(screen.getByTestId('add-scene-submit'))
    await waitFor(() => expect(calls).toHaveLength(1))
    expect(calls[0].path).toBe('/api/jobs/j1/scenes')
    expect(calls[0].body).toMatchObject({ position: 1, description: 'the lamp at night', setting: 'Atlantic coast', generate: true })

    const writing = { ...threeScenes, scenes: [scene(0, { writing_prompts: true, description: 'new' })] } as unknown as JobPublic
    show(writing)
    expect(screen.getByTestId('writing-prompts-0')).toBeInTheDocument()
  })

  it('keeps a scene’s unsaved edit on that scene when another is deleted', async () => {
    const calls: Call[] = []
    let current = threeScenes
    stubCalls(calls, () => current)
    const user = userEvent.setup()
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const { rerender } = render(<QueryClientProvider client={qc}><GatePanel job={current} assets={assets} /></QueryClientProvider>)
    // Three scenes stay expanded, so [1] is the second scene's keyframe prompt.
    await user.type(screen.getAllByDisplayValue('kf')[1], ' mine')
    expect(screen.getByDisplayValue('kf mine')).toBeInTheDocument()

    // The server deletes scene 1; the remaining scenes keep their ids but are renumbered.
    // Renumbered: the ids stay s1/s2, the positions become 0/1 — exactly what the server sends.
    current = { ...threeScenes, scenes: [scene(0, { id: 's1', description: 'second' }), scene(1, { id: 's2', description: 'third' })] } as unknown as JobPublic
    rerender(<QueryClientProvider client={qc}><GatePanel job={current} assets={assets} /></QueryClientProvider>)
    expect(screen.getByDisplayValue('kf mine')).toBeInTheDocument()   // still on the scene it was typed in
    expect(screen.getByTestId('scene-dirty-0')).toBeInTheDocument()   // which is now number 1
  })
})

describe('GatePanel render buttons', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('says what Approve renders and what a single scene renders', () => {
    const j = {
      ...job, face_swap: false, pending_gate: 'keyframes',
      profile: { video: { width: 832, height: 480, length: 65, workflow: 'animate' } },
      scenes: [
        scene(0, { keyframe_path: '/k0.png', video_path: '/c0.mp4' }),
        scene(1, { keyframe_path: '/k1.png' }),
        scene(2, { keyframe_path: '/k2.png' }),
      ],
    } as unknown as JobPublic
    render(<QueryClientProvider client={new QueryClient()}><GatePanel job={j} assets={assets} /></QueryClientProvider>)
    expect(screen.getByText(/the 2 scenes still missing a clip/)).toBeInTheDocument()
    // Scene 1 already has a clip: its button re-animates, and it names the cost of one scene.
    const reanimate = screen.getByText('Re-animate this scene')
    expect(reanimate).toBeInTheDocument()
    expect(reanimate.getAttribute('title')).toMatch(/clip only \(~.*\bmin\b.*\), whatever the gate/)
    // Two scenes without a clip get the button; the third mention is the header's explanation.
    expect(screen.getAllByRole('button', { name: 'Animate this scene' })).toHaveLength(2)
  })
})

describe('GatePanel version prompts', () => {
  afterEach(() => vi.unstubAllGlobals())

  const twoTakes = {
    ...job, face_swap: false,
    scenes: [scene(0, {
      keyframe_path: '/k2.png', keyframe_prompt: 'the current prompt',
      variants: [
        { id: 'v1', path: '/k1.png', url: '/k1.png', active: false, batch: 0, seed: 42, post_paths: [], source: 'generated', created_at: '' },
        { id: 'v2', path: '/k2.png', url: '/k2.png', active: true, batch: 1, seed: 53, post_paths: [], source: 'generated', created_at: '' },
      ],
    })],
  } as unknown as JobPublic

  it('shows what made an earlier take and puts its prompt back', async () => {
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const raw = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url
      const path = new URL(raw, 'http://localhost').pathname
      const body = path.endsWith('/versions/v1')
        ? { version_id: 'v1', kind: 'keyframe', path: '/k1.png', seed: 42, prompt: 'the prompt that made it',
            negative: 'waxy', prompt_from: 'record', source: 'generated', params: { steps: 24, cfg: 6 },
            backend: 'local', has_manifest: true, matches_scene_prompt: false, created_at: '', batch: 0 }
        : { error: { code: 'not_found', message: path } }
      return new Response(JSON.stringify(body), { status: 200, headers: { 'content-type': 'application/json' } })
    }))
    const user = userEvent.setup()
    render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <GatePanel job={twoTakes} assets={assets} /></QueryClientProvider>)

    await user.click(screen.getByTestId('version-info-keyframe-0-v1'))
    expect(await screen.findByText('the prompt that made it')).toBeInTheDocument()
    const panel = within(screen.getByTestId('version-detail-keyframe-0'))   // the tile label also says "seed 42"
    expect(panel.getByText(/seed 42/)).toBeInTheDocument()
    expect(panel.getByText(/24 steps/)).toBeInTheDocument()
    expect(panel.getByText(/Differs from the scene/)).toBeInTheDocument()

    await user.click(screen.getByTestId('version-restore-keyframe-0'))
    expect(screen.getByDisplayValue('the prompt that made it')).toBeInTheDocument()  // in the scene editor
    expect(screen.getByTestId('scene-dirty-0')).toBeInTheDocument()                  // unsaved, as any edit
  })
})

describe('GatePanel per-scene edit workflow', () => {
  afterEach(() => vi.unstubAllGlobals())

  const WORKFLOWS = {
    workflows: [
      { name: 'identity_edit', kind: 'edit', description: 'composition-locked', params: { width: {}, height: {}, negative: {} } },
      { name: 'kv_edit', kind: 'edit', description: 'keeps the image', params: { reference_megapixels: {} } },
    ],
    errors: {}, kinds: ['edit'], user_dir: '',
  }

  function stub(puts: unknown[]) {
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const raw = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url
      const url = new URL(raw, 'http://localhost')
      const method = init?.method ?? (input instanceof Request ? input.method : 'GET')
      if (url.pathname === '/api/workflows') {
        return new Response(JSON.stringify(WORKFLOWS), { status: 200, headers: { 'content-type': 'application/json' } })
      }
      // openapi-fetch may pass a Request rather than an init body.
      if (method !== 'GET') {
        puts.push(init?.body ? JSON.parse(String(init.body))
          : input instanceof Request ? await input.json().catch(() => null) : null)
      }
      return new Response(JSON.stringify(job), { status: 200, headers: { 'content-type': 'application/json' } })
    }))
  }

  const editable = {
    ...job, face_swap: false, profile: { edit_workflow: 'identity_edit' },
    scenes: [scene(0, { mode: 'edit', keyframe_path: '/k0.png', keyframe_url: '/media/k0.png', reference_asset_ids: ['room'] })],
  } as unknown as JobPublic

  it('lists the registered edit workflows and sends the chosen one', async () => {
    const puts: unknown[] = []
    stub(puts)
    const user = userEvent.setup()
    render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <GatePanel job={editable} assets={assets} /></QueryClientProvider>)

    const select = await screen.findByTestId('edit-workflow-0')
    expect(select).toHaveValue('')                                    // profile default first
    expect(screen.getByRole('option', { name: /profile default \(identity_edit\)/ })).toBeInTheDocument()
    expect(await screen.findByRole('option', { name: /kv_edit/ })).toBeInTheDocument()  // fetched
    expect(screen.queryByTestId('edit-base-0')).toBeNull()             // identity_edit sizes itself
    expect(screen.queryByTestId('edit-note-0')).toBeNull()

    await user.selectOptions(select, 'kv_edit')
    // An image-edit workflow starts from this scene's own frame, and says what it ignores.
    expect(screen.getByTestId('edit-base-0')).toBeChecked()
    expect(screen.getByTestId('edit-note-0')).toHaveTextContent('ignores the keyframe negative')

    await user.click(screen.getByText('Save changes'))
    await waitFor(() => expect(puts).toHaveLength(1))
    expect((puts[0] as { scenes: { edit_workflow: string; edit_base: string }[] }).scenes[0])
      .toMatchObject({ edit_workflow: 'kv_edit', edit_base: 'keyframe' })
  })

  it('keeps mode, the workflow selector and the references laid out as one row plus a block', async () => {
    // The row used to be a two-column grid [mode | references]; the full-width workflow block
    // pushed the references into the 8rem column, squashing them to 128px.
    stub([])
    render(<QueryClientProvider client={new QueryClient()}><GatePanel job={editable} assets={assets} /></QueryClientProvider>)
    const select = await screen.findByTestId('edit-workflow-0')
    const row = select.closest('div.flex')!                       // mode + "edit with" share a flex row
    const mode = screen.getAllByRole('combobox').find((el) => el !== select)!
    expect(row).toContainElement(mode)                            // they sit side by side
    // …and the references live outside that row, at full width.
    const refs = screen.getByTestId('refs-0-choose')
    expect(row.contains(refs)).toBe(false)
    expect(refs.closest('.grid')).toBeNull()
  })

  it('hides the workflow selector on a t2i scene', async () => {
    stub([])
    const t2i = { ...editable, scenes: [scene(0, { keyframe_path: '/k0.png' })] } as unknown as JobPublic
    render(<QueryClientProvider client={new QueryClient()}><GatePanel job={t2i} assets={assets} /></QueryClientProvider>)
    expect(screen.queryByTestId('edit-workflow-0')).toBeNull()
  })
})

describe('GatePanel with clips in the project', () => {
  afterEach(() => vi.unstubAllGlobals())

  const withClip = [...assets, asset('stock', 'video', 'drone over the coast', { name: 'stock.mp4' })]
  const j = {
    ...job, face_swap: false, status: 'done', pending_gate: null,
    scenes: [scene(0, { keyframe_path: '/k0.png', keyframe_url: '/media/k0.png' })],
  } as unknown as JobPublic

  it('offers a clip as a scene s video and never as an image', async () => {
    const posts: { url: string; body: Record<string, unknown> }[] = []
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const raw = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url
      const body = init?.body ? JSON.parse(String(init.body)) : input instanceof Request ? await input.json() : {}
      posts.push({ url: new URL(raw, 'http://localhost').pathname, body })
      return new Response(JSON.stringify(j), { status: 200, headers: { 'content-type': 'application/json' } })
    }))
    vi.stubGlobal('confirm', () => true)
    const user = userEvent.setup()
    render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <GatePanel job={j} assets={withClip} /></QueryClientProvider>)

    // The clip is not an image: it must not be offered where a keyframe is expected.
    await user.click(screen.getByTestId('use-image-0'))
    expect(screen.queryByTestId('keyframe-image-0-stock')).toBeNull()
    expect(screen.getByTestId('keyframe-image-0-room')).toBeInTheDocument()
    await user.click(screen.getByTestId('use-image-0'))

    await user.click(screen.getByTestId('use-video-0'))
    await user.click(screen.getByTestId('scene-video-0-stock'))
    await waitFor(() => expect(posts.some((p) => p.url.endsWith('/scenes/0/regenerate'))).toBe(true))
    expect(posts.find((p) => p.url.endsWith('/scenes/0/regenerate'))!.body)
      .toMatchObject({ what: 'video', clip_asset_id: 'stock' })
  })

  it('offers the control only when the project actually has a clip', () => {
    const only = (as: AssetOut[]) => {
      const { unmount } = render(
        <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
          <GatePanel job={j} assets={as} /></QueryClientProvider>)
      const there = !!screen.queryByTestId('use-video-0')
      unmount()
      return there
    }
    expect(only(withClip)).toBe(true)
    expect(only(assets)).toBe(false)     // images only: nothing to offer
  })
})

describe('GatePanel narration', () => {
  afterEach(() => vi.unstubAllGlobals())

  const spoken = {
    ...job, face_swap: false, status: 'done', pending_gate: null,
    profile: { video: { width: 832, height: 480, fps: 16, length: 49 } },
    scenes: [scene(0, {
      video_path: '/v0.mp4', video_url: '/media/v0.mp4',
      narration: 'Открий своя аромат.', narration_url: '/media/n0.wav',
      narration_seconds: 4.2, narration_stale: false,
    })],
  } as unknown as JobPublic

  it('shows the line against the clip it plays over, and warns when it runs long', () => {
    render(<QueryClientProvider client={new QueryClient()}><GatePanel job={spoken} assets={assets} /></QueryClientProvider>)
    // 49 frames at 16 fps = 3.06 s, so a 4.2 s line overflows.
    expect(screen.getByTestId('narration-fit-0')).toHaveTextContent('4.20s over a 3.06s clip')
    expect(screen.getByTestId('narration-fit-0')).toHaveTextContent('runs into the next scene')
    expect(screen.getByTestId('narration-audio-0')).toHaveAttribute('src', '/media/n0.wav')
    expect(screen.queryByTestId('narration-stale-0')).toBeNull()
  })

  it('marks a reworded line and speaks on request', async () => {
    const posts: string[] = []
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const raw = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url
      posts.push(new URL(raw, 'http://localhost').pathname)
      return new Response(JSON.stringify(spoken), { status: 200, headers: { 'content-type': 'application/json' } })
    }))
    const stale = { ...spoken, scenes: [{ ...spoken.scenes[0], narration_stale: true }] } as unknown as JobPublic
    render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <GatePanel job={stale} assets={assets} /></QueryClientProvider>)
    expect(screen.getByTestId('narration-stale-0')).toBeInTheDocument()

    await userEvent.click(screen.getByTestId('speak-0'))
    await waitFor(() => expect(posts.some((u) => u.endsWith('/scenes/0/speak'))).toBe(true))
  })

  it('speaks a line that was typed but not saved', async () => {
    // The button is enabled on what is in the box, so it must act on what is in the box. It used
    // to POST straight to /speak, and the server — which reads the SAVED scene — answered
    // "scene 1 has no line to speak" for a line plainly on screen.
    const calls: { method: string; path: string }[] = []
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const raw = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url
      const method = init?.method ?? (input instanceof Request ? input.method : 'GET')
      calls.push({ method, path: new URL(raw, 'http://localhost').pathname })
      return new Response(JSON.stringify(spoken), { status: 200, headers: { 'content-type': 'application/json' } })
    }))
    const blank = { ...spoken, scenes: [{ ...spoken.scenes[0], narration: '', narration_url: null }] } as unknown as JobPublic
    const user = userEvent.setup()
    render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <GatePanel job={blank} assets={assets} /></QueryClientProvider>)

    await user.type(screen.getByTestId('narration-0'), 'Открий своя аромат.')
    await user.click(screen.getByTestId('speak-0'))

    await waitFor(() => expect(calls.some((c) => c.path.endsWith('/speak'))).toBe(true))
    const put = calls.findIndex((c) => c.method === 'PUT' && c.path.endsWith('/scenes'))
    const spk = calls.findIndex((c) => c.path.endsWith('/speak'))
    expect(put).toBeGreaterThanOrEqual(0)
    expect(put).toBeLessThan(spk)      // saved first, then spoken
  })

  it('will not speak an empty line', () => {
    const silent = { ...spoken, scenes: [{ ...spoken.scenes[0], narration: '' }] } as unknown as JobPublic
    render(<QueryClientProvider client={new QueryClient()}><GatePanel job={silent} assets={assets} /></QueryClientProvider>)
    expect(screen.getByTestId('speak-0')).toBeDisabled()
  })
})
