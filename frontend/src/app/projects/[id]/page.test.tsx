import { Suspense } from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, describe, expect, it, vi } from 'vitest'
import ProjectPage from './page'
import type { AssetOut } from '@/api/types'

vi.mock('next/navigation', () => ({ useRouter: () => ({ push: vi.fn() }) }))

const asset = (id: string, extra: Partial<AssetOut>): AssetOut => ({
  id, kind: 'reference', name: `${id}.jpg`, label: '', label_source: '', note_caption: '', note_tags: [], note_text: '',
  describe_state: '', describe_error: '', sha256: '', created_at: '', path: '', url: `/api/projects/p1/assets/${id}/file`,
  thumb_url: `/api/projects/p1/assets/${id}/thumb`, ...extra,
})
const project = (assets: AssetOut[]) => ({
  id: 'p1', slug: 'bela', name: 'Bela', brand_notes: '', style_notes: '', default_profile_id: 'cinematic_16x9', export_root: '',
  export_root_effective: '/x', describe_hint: 'lighthouses', assets, created_at: '', updated_at: '', n_jobs: 0,
})

type Call = { method: string; path: string; body: unknown }
function stub(calls: Call[], assets: AssetOut[]) {
  vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const raw = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url
    const path = new URL(raw, 'http://localhost').pathname
    const method = init?.method ?? (input instanceof Request ? input.method : 'GET')
    const body = method === 'GET' ? null : init?.body ? JSON.parse(String(init.body)) : input instanceof Request ? await input.json() : null
    calls.push({ method, path, body })
    const json = (d: unknown, status = 200) => new Response(JSON.stringify(d), { status, headers: { 'content-type': 'application/json' } })
    if (path === '/api/projects/p1' && method === 'GET') return json(project(assets))
    if (path === '/api/profiles') return json([])
    if (path === '/api/jobs') return json({ items: [], total: 0, limit: 50, offset: 0 })
    if (path === '/api/projects/p1/assets/describe') return json({ queued: 2 }, 202)
    if (path.startsWith('/api/projects/p1/assets/')) return json(assets[0])
    return json({ error: { code: 'not_found', message: path } }, 404)
  }))
}

const renderPage = async () => {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  // A promise already marked fulfilled is read synchronously by React's use(), as Next hands pages their params.
  const params = Object.assign(Promise.resolve({ id: 'p1' }), { status: 'fulfilled', value: { id: 'p1' } })
  render(<QueryClientProvider client={qc}><Suspense fallback="…"><ProjectPage params={params} /></Suspense></QueryClientProvider>)
  await screen.findByText('Assets (3)')
}

const assets = [
  asset('plaque', { label: 'grey wall — a plaque', label_source: 'auto', note_text: 'MIZEN HEAD' }),
  asset('lamp', { label: 'Fresnel lens', label_source: 'human' }),
  asset('new', { describe_state: 'describing' }),
]

describe('project assets page', () => {
  afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks() })

  it('searches, filters to unreviewed labels and shows thumbnails', async () => {
    stub([], assets)
    const user = userEvent.setup()
    await renderPage()
    expect(screen.getByTestId('asset-count')).toHaveTextContent('3 of 3')
    expect(screen.getByAltText('plaque.jpg')).toHaveAttribute('src', '/api/projects/p1/assets/plaque/thumb')
    await user.type(screen.getByLabelText('search assets'), 'mizen')
    expect(screen.getByTestId('asset-count')).toHaveTextContent('1 of 3')
    await user.clear(screen.getByLabelText('search assets'))
    await user.click(screen.getByLabelText('unreviewed only'))
    expect(screen.getByTestId('asset-count')).toHaveTextContent('1 of 3')
    expect(screen.queryByTestId('describe-state-new')).toBeNull()  // the describing asset is not unreviewed
  })

  it('changes an asset kind and queues description', async () => {
    const calls: Call[] = []
    stub(calls, assets)
    const user = userEvent.setup()
    await renderPage()
    expect(screen.getByTestId('describe-state-new')).toHaveTextContent('Gemma is describing')
    await user.selectOptions(screen.getByLabelText('kind of lamp.jpg'), 'master_face')
    await waitFor(() => expect(calls.some((c) => c.method === 'PATCH')).toBe(true))
    expect(calls.find((c) => c.method === 'PATCH')).toMatchObject({ path: '/api/projects/p1/assets/lamp', body: { kind: 'master_face', label: null } })

    await user.click(screen.getByText('Describe unlabelled'))
    await waitFor(() => expect(screen.getByText('2 queued')).toBeInTheDocument())
    expect(calls.find((c) => c.path.endsWith('/describe'))!.body).toEqual({ asset_ids: null, force: false })

    await user.click(screen.getByText('describe again'))
    await waitFor(() => expect(calls.filter((c) => c.path.endsWith('/describe'))).toHaveLength(2))
    expect(calls.filter((c) => c.path.endsWith('/describe'))[1].body).toEqual({ asset_ids: ['plaque'], force: true })
  })
})
