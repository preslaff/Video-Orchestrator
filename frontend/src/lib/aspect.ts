/** Mirrors vorch/formats.py: presets + computed 16-aligned sizes. */
export const KEYFRAME_PRESETS: Record<string, [number, number]> = {
  '16:9': [1024, 576],
  '9:16': [576, 1024],
  '1:1': [768, 768],
  '4:5': [704, 880],
  '4:3': [896, 672],
  '3:4': [672, 896],
}
export const WAN_PRESETS: Record<string, [number, number]> = {
  '16:9': [832, 480],
  '9:16': [480, 832],
  '1:1': [640, 640],
  '4:5': [576, 720],
  '4:3': [640, 480],
  '3:4': [480, 640],
}
export const ASPECTS = Object.keys(KEYFRAME_PRESETS)

export function parseAspect(aspect: string): [number, number] | null {
  const m = /^(\d+):(\d+)$/.exec(aspect.trim())
  if (!m) return null
  const w = Number(m[1]),
    h = Number(m[2])
  return w > 0 && h > 0 ? [w, h] : null
}

const round16 = (x: number) => Math.max(16, Math.round(x / 16) * 16)

export function keyframeSize(aspect: string, megapixels = 0.6): [number, number] | null {
  if (KEYFRAME_PRESETS[aspect]) return KEYFRAME_PRESETS[aspect]
  const p = parseAspect(aspect)
  if (!p) return null
  const r = p[0] / p[1]
  const h = Math.sqrt((megapixels * 1_000_000) / r)
  return [round16(h * r), round16(h)]
}

export function wanSize(aspect: string, shortSide = 480): [number, number] | null {
  if (WAN_PRESETS[aspect]) return WAN_PRESETS[aspect]
  const p = parseAspect(aspect)
  if (!p) return null
  const r = p[0] / p[1]
  return r >= 1 ? [round16(shortSide * r), round16(shortSide)] : [round16(shortSide), round16(shortSide / r)]
}

export function orientation(aspect: string): 'landscape' | 'portrait' | 'square' {
  const p = parseAspect(aspect)
  if (!p || p[0] === p[1]) return 'square'
  return p[0] > p[1] ? 'landscape' : 'portrait'
}
