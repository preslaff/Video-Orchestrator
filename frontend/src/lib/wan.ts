/** Each animate workflow declares its own rules (`registry.SIZE_CONSTRAINTS` /
 *  `LENGTH_CONSTRAINTS`): Wan wants 4n+1 frames on 16-aligned sizes, LTX 8n+1 on 32-aligned
 *  ones. Sizes and lengths are snapped to the chosen model's rules before they are sent, or
 *  the profile is rejected at submit ("width must be a multiple of 32, got 720"). */
export const SIZE_MULTIPLE: Record<string, number> = {
  multiple_of_8: 8, multiple_of_16: 16, multiple_of_32: 32,
}

/** Nearest multiple of `m`, never below it.
 *
 *  Ties round **down**, to agree with `formats.round_to` — Python's round() is half-to-even, so
 *  720 at 32 is 704 there and would be 736 with JS's Math.round. Both are legal; differing from
 *  the backend is not. */
export const snapSize = (n: number, m: number, max = 1920): number =>
  Math.min(Math.max(m, Math.ceil(n / m - 0.5) * m), Math.floor(max / m) * m)

/** Nearest valid length for a "4n+1"/"8n+1" constraint (ties go up). */
export function snapLength(n: number, constraint = '4n+1'): number {
  const step = Number(constraint.split('n')[0]) || 4
  if (!Number.isFinite(n)) return step * 12 + 1
  return Math.max(1, step * Math.round((n - 1) / step) + 1)
}

/** Wan i2v only accepts length = 4n+1 (49, 81, 121 …). */
export const isWanLength = (n: number): boolean => Number.isInteger(n) && n >= 1 && n % 4 === 1

/** Nearest valid Wan length (ties go up). */
export function snapToWan(n: number): number {
  if (!Number.isFinite(n)) return 49
  const k = Math.round((n - 1) / 4)
  return Math.max(1, 4 * k + 1)
}

export const clipSeconds = (length: number, fps: number): number =>
  fps > 0 ? Math.round((length / fps) * 100) / 100 : 0

/** Aspect → (still, clip) sizes, mirroring `vorch/formats.py`'s presets.
 *
 *  Both are chosen together on purpose: the still is what an uploaded image is centre-cropped
 *  to, and a profile whose still and clip aspects differ is rejected at submit
 *  (`GenParams.validate`). 480 is the tested 16 GB class; 720 needs a cloud backend. */
export const ASPECTS = ['16:9', '9:16', '1:1', '4:5', '4:3', '3:4'] as const
export type Aspect = (typeof ASPECTS)[number]

const STILL: Record<Aspect, [number, number]> = {
  '16:9': [1024, 576], '9:16': [576, 1024], '1:1': [768, 768],
  '4:5': [704, 880], '4:3': [896, 672], '3:4': [672, 896],
}
const CLIP_480: Record<Aspect, [number, number]> = {
  '16:9': [832, 480], '9:16': [480, 832], '1:1': [640, 640],
  '4:5': [576, 720], '4:3': [640, 480], '3:4': [480, 640],
}
const CLIP_720: Record<Aspect, [number, number]> = {
  '16:9': [1280, 720], '9:16': [720, 1280], '1:1': [960, 960],
  '4:5': [720, 896], '4:3': [960, 720], '3:4': [720, 960],
}

export const stillSize = (aspect: Aspect): [number, number] => STILL[aspect]
export const clipSize = (aspect: Aspect, shortSide: 480 | 720): [number, number] =>
  (shortSide === 720 ? CLIP_720 : CLIP_480)[aspect]
