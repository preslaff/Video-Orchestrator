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
