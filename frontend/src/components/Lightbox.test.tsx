import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { LightboxProvider, ZoomButton } from './Lightbox'
import { MediaThumb } from './MediaThumb'
import { VersionStrip } from './VersionStrip'
import { AssetPicker } from './AssetPicker'
import type { AssetOut, ScenePublic } from '@/api/types'

const withBox = (ui: React.ReactNode) => render(<LightboxProvider>{ui}</LightboxProvider>)

describe('Lightbox', () => {
  it('opens an image preview as its original, with caption, and closes on Esc and ✕', async () => {
    const user = userEvent.setup()
    withBox(<MediaThumb url="/media/kf.png" alt="scene 1 keyframe" caption="Scene 1 · keyframe" />)
    expect(screen.queryByTestId('lightbox')).toBeNull()
    await user.click(screen.getByAltText('scene 1 keyframe'))
    expect(screen.getByTestId('lightbox-image')).toHaveAttribute('src', '/media/kf.png')
    expect(screen.getByText('Scene 1 · keyframe')).toBeInTheDocument()
    await user.keyboard('{Escape}')
    expect(screen.queryByTestId('lightbox')).toBeNull()
    await user.click(screen.getByAltText('scene 1 keyframe'))
    await user.click(screen.getByTestId('lightbox-close'))
    expect(screen.queryByTestId('lightbox')).toBeNull()
  })

  it('opens a video preview from its ⤢ as a playing video', async () => {
    const user = userEvent.setup()
    const pause = vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {})
    withBox(<MediaThumb url="/media/clip.mp4" video caption="Scene 2 · clip" />)
    await user.click(screen.getByTestId('enlarge-video'))
    expect(screen.getByTestId('lightbox-video')).toHaveAttribute('src', '/media/clip.mp4')
    expect(pause).toHaveBeenCalled()                                   // the small player stops
    pause.mockRestore()
  })

  it('steps through a group with arrows and stops at the ends', async () => {
    const user = userEvent.setup()
    const items = [{ url: '/a.png', caption: 'a' }, { url: '/b.png', caption: 'b' }, { url: '/c.png', caption: 'c' }]
    withBox(<div className="relative"><ZoomButton items={items} index={1} testId="z" /></div>)
    await user.click(screen.getByTestId('z'))
    expect(screen.getByTestId('lightbox-counter')).toHaveTextContent('2 / 3')
    await user.keyboard('{ArrowRight}')
    expect(screen.getByTestId('lightbox-image')).toHaveAttribute('src', '/c.png')
    expect(screen.getByTestId('lightbox-next')).toBeDisabled()
    await user.keyboard('{ArrowRight}')                                 // the end: stays
    expect(screen.getByTestId('lightbox-counter')).toHaveTextContent('3 / 3')
    await user.click(screen.getByTestId('lightbox-prev'))
    await user.click(screen.getByTestId('lightbox-prev'))
    expect(screen.getByTestId('lightbox-image')).toHaveAttribute('src', '/a.png')
  })

  it('⤢ on a version tile opens the clip without selecting the version', async () => {
    const user = userEvent.setup()
    const onSelect = vi.fn()
    const scene = {
      index: 0, variants: [{ id: 'v1', path: '/k.png', url: '/k.png', active: true, batch: 0 }],
      clips: [
        { id: 'c1', url: '/c1.mp4', keyframe_path: '/k.png', active: true, batch: 0, seed: 1 },
        { id: 'c2', url: '/c2.mp4', keyframe_path: '/k.png', active: false, batch: 1, seed: 2 },
      ],
    } as unknown as ScenePublic
    withBox(<VersionStrip scene={scene} kind="clip" onSelect={onSelect} onDelete={() => {}} />)
    await user.click(screen.getByTestId('version-zoom-clip-0-c1'))
    expect(screen.getByTestId('lightbox-video')).toHaveAttribute('src', '/c1.mp4')
    expect(onSelect).not.toHaveBeenCalled()
  })

  it('⤢ on a picker tile shows the original, not the thumbnail, and keeps the selection', async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()
    const a = { id: 'x', kind: 'reference', name: 'x.jpg', label: 'lens', label_source: 'human', note_caption: 'Fresnel lens',
      note_tags: [], note_text: '', describe_state: '', describe_error: '', thumb_url: '/thumb/x.jpg', sha256: '', created_at: '',
      path: '/x.jpg', url: '/file/x.jpg' } as unknown as AssetOut
    withBox(<AssetPicker assets={[a]} selected={[]} onChange={onChange} testId="p" />)
    await user.click(screen.getByTestId('p-x-zoom'))
    expect(screen.getByTestId('lightbox-image')).toHaveAttribute('src', '/file/x.jpg')
    expect(screen.getByText('lens — Fresnel lens')).toBeInTheDocument()
    expect(onChange).not.toHaveBeenCalled()
  })
})
