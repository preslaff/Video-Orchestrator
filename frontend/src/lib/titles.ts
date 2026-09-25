import type { TextItem } from '@/api/types'

/** What the server offers when the status has not loaded yet. The real list is `text_effects`. */
export const FALLBACK_EFFECTS = ['none', 'fade', 'slide', 'pop', 'typewriter']

/** A new caption, matching the server's own defaults so an unedited item round-trips unchanged. */
export function blankText(patch: Partial<TextItem> = {}): TextItem {
  return {
    text: '', start: 0, duration: 2, effect: 'fade', effect_seconds: 0.35,
    font: 'DejaVu Sans', size: 0.048, colour: 'FFFFFF', outline: '202020',
    outline_width: 0.004, shadow: 0, bold: true, italic: false, align: 2,
    margin_x: 0.08, margin_y: 0.08, angle: 0, opacity: 1, ...patch,
  }
}

/** ASS numpad anchors, as a person reads them. 1 is bottom-left, 9 is top-right. */
export const ANCHORS: { value: number; label: string }[] = [
  { value: 7, label: 'top left' }, { value: 8, label: 'top' }, { value: 9, label: 'top right' },
  { value: 4, label: 'left' }, { value: 5, label: 'centre' }, { value: 6, label: 'right' },
  { value: 1, label: 'bottom left' }, { value: 2, label: 'bottom' }, { value: 3, label: 'bottom right' },
]

/**
 * Where a caption sits in a preview box, as CSS.
 *
 * Mirrors `titles._anchor`: the alignment names which corner of the text is placed, and the
 * margins are fractions of the box — so the preview scales with the element exactly as libass
 * scales with PlayRes, and the two cannot drift when the delivery size changes.
 */
export function anchorStyle(item: TextItem): React.CSSProperties {
  const col = (item.align! - 1) % 3        // 0 left, 1 centre, 2 right
  const row = Math.floor((item.align! - 1) / 3)  // 0 bottom, 1 middle, 2 top
  const mx = `${(item.margin_x ?? 0.08) * 100}%`
  const my = `${(item.margin_y ?? 0.08) * 100}%`
  const s: React.CSSProperties = { position: 'absolute', textAlign: ['left', 'center', 'right'][col] as 'left' }
  if (col === 0) { s.left = mx } else if (col === 2) { s.right = mx } else { s.left = mx; s.right = mx }
  if (row === 0) { s.bottom = my } else if (row === 2) { s.top = my } else { s.top = '50%'; s.transform = 'translateY(-50%)' }
  return s
}

/** Type size and weight for a preview box `px` tall — the same fraction the renderer uses. */
export function textStyle(item: TextItem, boxHeight: number): React.CSSProperties {
  const px = Math.max(8, (item.size ?? 0.048) * boxHeight)
  const ow = Math.max(0.5, (item.outline_width ?? 0.004) * boxHeight)
  return {
    fontFamily: `'${item.font || 'DejaVu Sans'}', sans-serif`,
    fontSize: `${px}px`,
    lineHeight: 1.2,
    fontWeight: item.bold ? 700 : 400,
    fontStyle: item.italic ? 'italic' : 'normal',
    color: `#${item.colour || 'FFFFFF'}`,
    opacity: item.opacity ?? 1,
    // An approximation of the ASS outline; libass draws a true stroke, the browser fakes one.
    textShadow: [
      `${ow}px 0 0 #${item.outline}`, `-${ow}px 0 0 #${item.outline}`,
      `0 ${ow}px 0 #${item.outline}`, `0 -${ow}px 0 #${item.outline}`,
    ].join(', '),
    whiteSpace: 'pre-line',
  }
}

/** Is this caption on screen at `t` seconds? The timing is exact — it is plain arithmetic. */
export function visibleAt(item: TextItem, t: number): boolean {
  const start = item.start ?? 0
  return t >= start && t < start + (item.duration ?? 0)
}

/**
 * Where a scene's caption lands on the finished film.
 *
 * The mirror of `assembly._texts_for`, and it must agree with it to the frame: `at` is the scene's
 * own offset from `media.item_offsets`, and both the start and the duration divide by the clip's
 * speed so the caption stays glued to the frames it annotates.
 */
export function placeOnTimeline(item: TextItem, at: number, speed = 1): { start: number; duration: number } {
  const s = speed || 1
  return { start: at + (item.start ?? 0) / s, duration: (item.duration ?? 0) / s }
}
