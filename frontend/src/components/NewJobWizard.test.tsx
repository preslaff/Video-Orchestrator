import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { NewJobWizard } from './NewJobWizard'

const push = vi.fn()
vi.mock('next/navigation', () => ({ useRouter: () => ({ push }) }))

const PROFILE = {
  id: 'cinematic_16x9', name: 'Cinematic 16:9', purpose: 'cinematic', kind: 'video', aspect: '16:9',
  keyframe: { workflow: 'keyframe', width: 1024, height: 576, steps: 24, cfg: 5, unet: null },
  video: { workflow: 'animate', width: 832, height: 480, length: 49, fps: 16, negative: '' },
  stills: { variants: 1 }, style: { keyframe_prefix: '', keyframe_negative: '', animation_suffix: 'cinematic', animation_negative: '' },
  face_swap: { enabled: false, workflow: 'faceswap', restore_model: 'x' }, evaluate: { enabled: true, max_attempts: 2, review_threshold: 4 },
  edit_workflow: 'identity_edit', post_steps: [], builtin: true, created_at: '', updated_at: '',
}

function mockFetch(env: { ok: boolean; checks: { name: string; status: string; detail: string; data: object }[] }) {
  const posts: { url: string; body: unknown }[] = []
  vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const raw = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url
    const url = new URL(raw, 'http://localhost').pathname
    const method = init?.method ?? (input instanceof Request ? input.method : 'GET')
    const readBody = async () => (init?.body ? JSON.parse(String(init.body)) : input instanceof Request ? await input.json() : null)
    const json = (data: unknown, status = 200) => new Response(JSON.stringify(data), { status, headers: { 'content-type': 'application/json' } })
    if (url.startsWith('/api/projects')) return json([])
    if (url.startsWith('/api/profiles')) return json([PROFILE])
    if (url.startsWith('/api/setup/env')) return json({ ...env, status: env.ok ? 'ok' : 'fail', checked_at: '', duration_ms: 1, profile_id: null, deep: false })
    if (url === '/api/jobs' && method === 'POST') {
      posts.push({ url, body: await readBody() })
      return json({ id: 'job-1', status: 'pending', scenes: [] })
    }
    return json({ error: { code: 'not_found', message: url } }, 404)
  }))
  return posts
}

function renderWizard() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(<QueryClientProvider client={qc}><NewJobWizard /></QueryClientProvider>)
}

describe('NewJobWizard', () => {
  beforeEach(() => push.mockReset())
  afterEach(() => vi.unstubAllGlobals())

  it('blocks submit without an idea, then on env failures until forced, and posts the spec', async () => {
    const posts = mockFetch({ ok: false, checks: [{ name: 'gemma', status: 'fail', detail: 'unreachable', data: {} }] })
    const user = userEvent.setup()
    renderWizard()
    await user.click(screen.getByText('Next')) // project -> profile
    await screen.findByText('Cinematic 16:9')
    await user.click(screen.getByText('Next')) // profile -> brief
    await user.click(screen.getByText('Next')) // brief -> review (idea empty)
    expect(await screen.findByText('An idea is required.')).toBeInTheDocument()
    expect(screen.getByTestId('submit')).toBeDisabled()

    await user.click(screen.getByText('3. Brief'))
    await user.type(screen.getByTestId('idea'), 'a cat on a roof')
    await user.click(screen.getByText('Next'))
    expect(await screen.findByText('unreachable')).toBeInTheDocument() // env failure listed
    expect(screen.getByTestId('submit')).toBeDisabled()

    await user.click(screen.getByTestId('force'))
    expect(screen.getByTestId('submit')).toBeEnabled()
    await user.click(screen.getByTestId('submit'))
    await waitFor(() => expect(push).toHaveBeenCalledWith('/jobs/job-1'))
    expect(posts).toHaveLength(1)
    expect(posts[0].body).toMatchObject({ idea: 'a cat on a roof', force: true, mode: 't2i', profile_id: null, face_swap: null })
  })

  it('submits directly when the environment is fine', async () => {
    const posts = mockFetch({ ok: true, checks: [{ name: 'comfy', status: 'ok', detail: '', data: {} }] })
    const user = userEvent.setup()
    renderWizard()
    await user.click(screen.getByText('Next'))
    await screen.findByText('Cinematic 16:9')
    await user.click(screen.getByText('Next'))
    await user.type(screen.getByTestId('idea'), 'bread at dawn')
    await user.click(screen.getByText('Next'))
    await screen.findByText('Environment OK.')
    await user.click(screen.getByTestId('submit'))
    await waitFor(() => expect(posts).toHaveLength(1))
    expect(posts[0].body).toMatchObject({ idea: 'bread at dawn', force: false })
  })

  it('shows what the job will cost in GPU time, multiplied by the scene count', async () => {
    mockFetch({ ok: true, checks: [{ name: 'comfy', status: 'ok', detail: '', data: {} }] })
    const user = userEvent.setup()
    renderWizard()
    await user.click(screen.getByText('Next'))
    await screen.findByText('Cinematic 16:9')
    await user.click(screen.getByText('Next'))
    await user.type(screen.getByTestId('idea'), 'bread at dawn')
    await user.type(screen.getByRole('spinbutton', { name: /scenes/i }), '6')
    await user.click(screen.getByText('Next'))
    await screen.findByText('Environment OK.')
    // 832×480×49 on this GPU ≈ 7.9 min a clip, so six scenes is most of an hour
    expect(screen.getByTestId('job-cost')).toHaveTextContent(/per clip on this GPU/)
    expect(screen.getByTestId('job-cost')).toHaveTextContent(/for 6 scenes/)
  })

  const user = userEvent.setup()

  it('preselects the project from initialProjectId and starts at the profile step', async () => {
    const project = { id: 'p1', slug: 'p1', name: 'Perfume', default_profile_id: 'cinematic_16x9', n_assets: 0, n_jobs: 0, updated_at: '' }
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const raw = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url
      const url = new URL(raw, 'http://localhost').pathname
      const json = (data: unknown, status = 200) => new Response(JSON.stringify(data), { status, headers: { 'content-type': 'application/json' } })
      if (url === '/api/projects') return json([project])
      if (url === '/api/projects/p1') return json({ ...project, brand_notes: '', style_notes: '', export_root: '', export_root_effective: '/x', assets: [], created_at: '' })
      if (url.startsWith('/api/profiles')) return json([PROFILE])
      return json({ error: { code: 'not_found', message: url } }, 404)
    }))
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(<QueryClientProvider client={qc}><NewJobWizard initialProjectId="p1" /></QueryClientProvider>)
    expect(await screen.findByText('Cinematic 16:9')).toBeInTheDocument()   // profile step is shown first
    await user.click(screen.getByText('1. Project'))
    await waitFor(() => expect((screen.getByTestId('project') as HTMLSelectElement).value).toBe('p1'))
  })
})

const ASSET = (id: string, label: string) => ({
  id, kind: 'product', name: `${id}.png`, label, label_source: 'human', note_caption: 'a flacon',
  note_tags: [], note_text: '', describe_state: '', describe_error: '', sha256: '', created_at: '',
  path: `/a/${id}.png`, url: `/file/${id}.png`, thumb_url: `/thumb/${id}.png`,
})

describe('NewJobWizard animating existing images', () => {
  beforeEach(() => push.mockReset())
  afterEach(() => vi.unstubAllGlobals())

  /** Like mockFetch, but with a project that actually holds assets to animate. */
  function mockWithAssets() {
    const posts: { url: string; body: unknown }[] = []
    const project = { id: 'p1', slug: 'perfume', name: 'Perfume', brand_notes: '', style_notes: '',
      default_profile_id: 'cinematic_16x9', export_root: '', export_root_effective: '/x',
      describe_hint: '', assets: [ASSET('a', 'stone'), ASSET('b', 'silk')], created_at: '',
      updated_at: '', n_jobs: 0 }
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const raw = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url
      const url = new URL(raw, 'http://localhost').pathname
      const method = init?.method ?? (input instanceof Request ? input.method : 'GET')
      const json = (data: unknown) => new Response(JSON.stringify(data), { status: 200, headers: { 'content-type': 'application/json' } })
      if (url === '/api/projects') return json([{ ...project, n_assets: 2, last_job_at: null }])
      if (url === '/api/projects/p1') return json(project)
      if (url.startsWith('/api/profiles')) return json([PROFILE])
      if (url.startsWith('/api/setup/env')) return json({ ok: true, status: 'ok', checks: [], checked_at: '', duration_ms: 1, profile_id: null, deep: false })
      if (url === '/api/jobs' && method === 'POST') {
        posts.push({ url, body: init?.body ? JSON.parse(String(init.body)) : await (input as Request).json() })
        return json({ id: 'job-2', status: 'pending', scenes: [] })
      }
      return json({ error: { code: 'not_found', message: url } })
    }))
    return posts
  }

  it('asks for images instead of an idea and posts them in click order', async () => {
    const posts = mockWithAssets()
    const user = userEvent.setup()
    renderWizard()
    await screen.findByRole('option', { name: /Perfume/ })      // the project list has arrived
    await user.selectOptions(screen.getByTestId('project'), 'p1')
    await user.click(screen.getByText('Next'))          // project -> profile
    await screen.findByText('Cinematic 16:9')
    await user.click(screen.getByText('Next'))          // profile -> brief

    await user.click(screen.getByTestId('source-animate'))
    expect(screen.queryByTestId('mode')).toBeNull()      // storyboard-only fields step aside
    expect(screen.getByText(/Direction \(tone and pace/)).toBeInTheDocument()

    await user.click(await screen.findByTestId('animate-b'))   // click order = play order
    await user.click(screen.getByTestId('animate-a'))
    await user.click(screen.getByText('Next'))
    await user.click(screen.getByTestId('submit'))

    await waitFor(() => expect(posts).toHaveLength(1))
    const body = posts[0].body as { keyframe_asset_ids: string[]; reference_asset_ids: string[]; idea: string }
    expect(body.keyframe_asset_ids).toEqual(['b', 'a'])
    expect(body.reference_asset_ids).toEqual([])
  })

  it('will not animate without images, or with a face swap on', async () => {
    mockWithAssets()
    const user = userEvent.setup()
    renderWizard()
    await screen.findByRole('option', { name: /Perfume/ })      // the project list has arrived
    await user.selectOptions(screen.getByTestId('project'), 'p1')
    await user.click(screen.getByText('Next'))
    await screen.findByText('Cinematic 16:9')
    await user.click(screen.getByText('Next'))
    await user.click(screen.getByTestId('source-animate'))
    await user.click(screen.getByText('Next'))
    expect(await screen.findByText('Pick at least one image to animate.')).toBeInTheDocument()

    await user.click(screen.getByText('3. Brief'))
    await user.selectOptions(screen.getByTestId('face_swap'), 'on')
    await user.click(screen.getByText('Next'))
    expect(await screen.findByText(/would paste a face onto your images/)).toBeInTheDocument()
  })
})

describe('NewJobWizard output overrides', () => {
  beforeEach(() => push.mockReset())
  afterEach(() => vi.unstubAllGlobals())

  it('sends the chosen model and size as profile overrides, still and clip together', async () => {
    const posts = mockFetch({ ok: true, checks: [] })
    const user = userEvent.setup()
    renderWizard()
    await user.click(screen.getByText('Next'))
    await screen.findByText('Cinematic 16:9')
    await user.click(screen.getByText('Next'))
    await user.type(screen.getByTestId('idea'), 'a teaser')

    // Default: no overrides at all, and the summary reflects the profile.
    expect(screen.getByTestId('output-summary')).toHaveTextContent('still 1024×576 → clip 832×480')

    await user.selectOptions(screen.getByTestId('clip-aspect'), '9:16')
    await user.selectOptions(screen.getByTestId('clip-class'), '720')
    await user.clear(screen.getByTestId('clip-length'))
    await user.type(screen.getByTestId('clip-length'), '81')
    expect(screen.getByTestId('output-summary')).toHaveTextContent('still 576×1024 → clip 720×1280')
    expect(screen.getByTestId('output-summary')).toHaveTextContent('81 frames @ 16 fps (~5.06s)')

    await user.click(screen.getByText('Next'))
    await user.click(screen.getByTestId('submit'))
    await waitFor(() => expect(posts).toHaveLength(1))
    const body = posts[0].body as { profile_overrides: { aspect: string; keyframe: object; video: object } }
    // The still moves with the clip: a profile whose aspects differ is rejected at submit.
    expect(body.profile_overrides).toEqual({
      aspect: '9:16',
      keyframe: { width: 576, height: 1024 },
      video: { length: 81, width: 720, height: 1280 },
    })
  })

  it('leaves the profile alone when nothing is chosen', async () => {
    const posts = mockFetch({ ok: true, checks: [] })
    const user = userEvent.setup()
    renderWizard()
    await user.click(screen.getByText('Next'))
    await screen.findByText('Cinematic 16:9')
    await user.click(screen.getByText('Next'))
    await user.type(screen.getByTestId('idea'), 'x')
    await user.click(screen.getByText('Next'))
    await user.click(screen.getByTestId('submit'))
    await waitFor(() => expect(posts).toHaveLength(1))
    expect((posts[0].body as { profile_overrides: unknown }).profile_overrides).toBeNull()
  })
})
