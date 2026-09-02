/** SSE contract with the backend (vorch/api/events.py). `id` is per-job and monotonic. */
export const EVENT_TYPES = [
  'status',
  'phase',
  'step',
  'storyboard',
  'verdict',
  'keyframe',
  'variant',
  'video',
  'postprocess',
  'stills',
  'gate',
  'export',
  'error',
  'end',
] as const

export type EventType = (typeof EVENT_TYPES)[number]

export interface JobEvent {
  id?: number
  type: EventType | string
  job_id: string
  /** Client-side arrival time (ms); set by useJobEvents for elapsed-time display. */
  _at?: number
  [key: string]: unknown
}

export const PHASE_LABELS: Record<string, string> = {
  storyboard: 'Storyboard',
  keyframes: 'Keyframes',
  videos: 'Videos',
  stills: 'Finishing stills',
  export: 'Export',
}

/** One-line description of a running step, for the activity indicator. */
export function describeStep(e: JobEvent): string {
  const scene = typeof e.index === 'number' ? `scene ${Number(e.index) + 1}` : ''
  const variant = typeof e.variant === 'number' && Number(e.variants) > 1 ? ` · variant ${Number(e.variant) + 1}/${String(e.variants)}` : ''
  switch (e.step) {
    case 'storyboard':
      return 'Gemma is writing the storyboard and the per-scene prompts'
    case 'keyframe': {
      const attempt = Number(e.max_attempts) > 1 ? ` · attempt ${String(e.attempt)}/${String(e.max_attempts)}` : ''
      const mode = e.mode === 'edit' ? ' (reference edit)' : ''
      return `Generating keyframe for ${scene}${variant}${attempt}${mode}${Number(e.max_attempts) > 1 ? ' — Gemma evaluates each attempt' : ''}`
    }
    case 'swap':
      return `Imposing the master face on ${scene}${variant} (ReActor)`
    case 'animate':
      return `Animating ${scene} with Wan — ${String(e.length)} frames @ ${String(e.fps)} fps (~${String(e.seconds)}s); this is the slow step`
    case 'post':
      return `Post-processing ${scene}${variant}: ${Array.isArray(e.chain) ? e.chain.join(' → ') : ''}`
    default:
      return String(e.step ?? 'working')
  }
}

/** Events after which the jobs list (status column) should refresh. */
export const LIST_EVENTS = new Set<string>(['status', 'gate', 'end', 'error', 'export'])

export function describeEvent(e: JobEvent): string {
  switch (e.type) {
    case 'status':
      return `status → ${String(e.status)}`
    case 'phase':
      return `phase: ${PHASE_LABELS[String(e.phase)] ?? String(e.phase)}${e.total ? ` (${String(e.total)} scene(s))` : ''}`
    case 'step':
      return describeStep(e)
    case 'storyboard':
      return `storyboard: ${String(e.scenes)} scene(s)`
    case 'verdict':
      return `scene ${String(e.index)} attempt ${String(e.attempt)}: ${String(e.decision)} (score ${String(e.score)})`
    case 'keyframe':
      return `scene ${String(e.index)}: keyframe ready`
    case 'variant':
      return `scene ${String(e.index)}: variant ${Number(e.variant) + 1} ready`
    case 'video':
      return `scene ${String(e.index)}: clip ready`
    case 'postprocess':
      return `scene ${String(e.index)}: post-step ${Number(e.step) + 1} done`
    case 'stills':
      return `scene ${String(e.index)}: ${String(e.variants)} still(s) finished`
    case 'gate':
      return `paused at gate "${String(e.gate)}" — approval needed`
    case 'export':
      return `exported ${Array.isArray(e.files) ? e.files.length : '?'} file(s) → ${String(e.path)}`
    case 'error':
      return `error: ${String(e.error)}`
    case 'end':
      return 'run segment finished'
    default:
      return e.type
  }
}
