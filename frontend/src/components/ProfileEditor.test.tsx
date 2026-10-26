import { fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { ProfileEditor } from './ProfileEditor'
import type { OutputProfile } from '@/api/types'

vi.mock('next/navigation', () => ({ useRouter: () => ({ push: vi.fn() }) }))

const profile = {
  id: 'social_9x16', name: 'Social vertical', purpose: 'social_vertical', kind: 'video', aspect: '9:16',
  keyframe: { workflow: 'keyframe', width: 576, height: 1024, steps: 24, cfg: 5, unet: null },
  video: { workflow: 'animate', width: 480, height: 832, length: 49, fps: 16, negative: '' },
  stills: { variants: 1 }, style: { keyframe_prefix: '', keyframe_negative: '', animation_suffix: '', animation_negative: '' },
  face_swap: { enabled: false, workflow: 'faceswap', restore_model: 'x' },
  evaluate: { enabled: true, max_attempts: 2, review_threshold: 4 },
  edit_workflow: 'identity_edit', post_steps: [], builtin: false, created_at: '', updated_at: '',
} as unknown as OutputProfile

const WORKFLOWS = {
  workflows: [{ name: 'animate', kind: 'animate' }, { name: 'keyframe', kind: 'keyframe' },
              { name: 'keyframe_flux3', kind: 'keyframe', params: { steps: { default: 16 }, cfg: { default: 1.0 } } },
              { name: 'api_wan_t2i', kind: 'keyframe', params: {} },
              { name: 'identity_edit', kind: 'edit' }, { name: 'faceswap', kind: 'faceswap' }],
  errors: {}, kinds: ['keyframe', 'animate'], user_dir: '~/vorch/workflows',
}

function renderEditor() {
  vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
    const raw = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url
    const body = new URL(raw, 'http://localhost').pathname.startsWith('/api/workflows') ? WORKFLOWS : []
    return new Response(JSON.stringify(body), { status: 200, headers: { 'content-type': 'application/json' } })
  }))
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(<QueryClientProvider client={qc}><ProfileEditor profile={profile} mode="edit" /></QueryClientProvider>)
}

describe('ProfileEditor size fields', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('lets a size be cleared and retyped instead of snapping back to a default', async () => {
    renderEditor()
    const user = userEvent.setup()
    const widths = screen.getAllByLabelText('width')
    const clipWidth = widths[widths.length - 1]

    await user.clear(clipWidth)
    expect(clipWidth).toHaveValue(null)                       // stays empty, does not restore 480
    expect(screen.getByText(/Fill in: clip size/)).toBeInTheDocument()

    await user.type(clipWidth, '832')
    expect(clipWidth).toHaveValue(832)                        // not 480832
    expect(screen.queryByText(/Fill in:/)).toBeNull()
  })

  it('does not snap a half-typed clip length to 4n+1 on blur', async () => {
    renderEditor()
    const user = userEvent.setup()
    const length = screen.getByLabelText(/length/)
    await user.clear(length)
    await user.tab()
    expect(length).toHaveValue(null)                          // an empty field is not "1"
    expect(screen.getByText(/Fill in: clip length\/fps/)).toBeInTheDocument()
  })
})

describe('ProfileEditor render cost', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('shows the time per clip and the VRAM headroom for the chosen size', () => {
    renderEditor()
    const cost = screen.getByTestId('render-cost')
    expect(cost).toHaveTextContent(/per clip on this GPU/)
    expect(cost).toHaveTextContent(/1\.0× the tested 16 GB budget/)
    expect(cost).not.toHaveTextContent(/OOM/)
  })

  it('warns when a custom size leaves the tested budget', async () => {
    renderEditor()
    const widths = screen.getAllByLabelText('width')
    const heights = screen.getAllByLabelText('height')
    fireEvent.change(widths[widths.length - 1], { target: { value: '832' } })      // the clip, 832×1248
    fireEvent.change(heights[heights.length - 1], { target: { value: '1248' } })
    const cost = screen.getByTestId('render-cost')
    expect(cost).toHaveTextContent(/2\.6× the tested 16 GB budget/)
    expect(cost).toHaveTextContent(/expect RAM offload, long renders or OOM/)
    expect(cost).toHaveTextContent(/~\d+–\d+ min per clip/)   // the wait that number costs
  })
})

describe('ProfileEditor keyframe workflow', () => {
  afterEach(() => vi.unstubAllGlobals())

  it("switching model applies that workflow's own steps and cfg", async () => {
    renderEditor()
    const user = userEvent.setup()
    const pick = screen.getAllByLabelText('workflow')[0]
    await screen.findByRole('option', { name: 'keyframe_flux3' })
    await user.selectOptions(pick, 'keyframe_flux3')
    expect(screen.getByLabelText('steps')).toHaveValue(16)
    expect(screen.getByLabelText('cfg')).toHaveValue(1)

    await user.selectOptions(pick, 'api_wan_t2i')              // declares none: keep what is there
    expect(screen.getByLabelText('steps')).toHaveValue(16)
    expect(screen.getByLabelText('cfg')).toHaveValue(1)
  })
})
