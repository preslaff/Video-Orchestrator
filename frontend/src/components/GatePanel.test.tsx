import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { GOTO_SCENE, GatePanel } from './GatePanel'
import type { AssetOut, JobPublic } from '@/api/types'

const scene = (index: number, extra: Partial<JobPublic['scenes'][number]> = {}) => ({
  index, id: `s${index}`, writing_prompts: false, description: 'd', character: '', setting: '', mood: '', continuous: false, status: 'pending',
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
      calls.push({ method, path: url.pathname, body, query: url.search })
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
