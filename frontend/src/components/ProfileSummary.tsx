import Link from 'next/link'
import type { JobPublic } from '@/api/types'
import { useWorkflows } from '@/hooks/useSetup'

type Snap = {
  id?: string; name?: string; purpose?: string; kind?: string; aspect?: string
  keyframe?: { workflow?: string; width?: number; height?: number; steps?: number; cfg?: number }
  video?: { workflow?: string; width?: number; height?: number; length?: number; fps?: number } | null
  stills?: { variants?: number }
  style?: { keyframe_prefix?: string; keyframe_negative?: string; animation_suffix?: string; animation_negative?: string }
  face_swap?: { enabled?: boolean; workflow?: string; restore_model?: string }
  evaluate?: { enabled?: boolean; max_attempts?: number; review_threshold?: number }
  edit_workflow?: string
  post_steps?: { workflow: string; params?: Record<string, unknown> }[]
  builtin?: boolean
}

/** The profile a job was run with — its frozen snapshot (`Job.profile`), not the live profile,
 * so the settings shown are exactly the ones that produced the files. */
export function ProfileSummary({ job }: { job: JobPublic }) {
  const snap = job.profile as Snap | null
  const wfs = useWorkflows()
  const backendOf = (name?: string) => wfs.data?.workflows.find((w) => w.name === name)?.backend
  const wf = (name?: string) => {
    if (!name) return '—'
    const b = backendOf(name)
    return (
      <span className="font-mono text-xs">{name}{b && b !== 'local' ? <span className="ml-1 rounded bg-sky-100 px-1 text-sky-800">{b}</span> : null}</span>
    )
  }
  if (!snap) return <p className="text-sm text-neutral-500">This job predates profile snapshots (run-config: face swap {job.face_swap ? 'on' : 'off'}, evaluate {job.evaluate ? 'on' : 'off'}).</p>
  const kf = snap.keyframe ?? {}
  const vid = snap.video ?? null
  const rows: [string, React.ReactNode][] = [
    ['profile', <span key="p">{snap.name ?? snap.id} <span className="font-mono text-xs text-neutral-500">{snap.id}</span>{snap.builtin ? <span className="ml-1 text-xs text-neutral-500">built-in</span> : null}{snap.id && !snap.builtin ? <Link href="/setup/profiles" className="ml-2 text-xs hover:underline">edit profiles</Link> : null}</span>],
    ['format', `${snap.kind ?? 'video'} · ${snap.aspect ?? '16:9'} · ${snap.purpose ?? ''}`],
    ['keyframe', <span key="k">{wf(kf.workflow)} · {kf.width}×{kf.height} · {kf.steps} steps · cfg {kf.cfg}</span>],
    ...(vid ? [['video', <span key="v">{wf(vid.workflow)} · {vid.width}×{vid.height} · {vid.length} frames @ {vid.fps} fps ({vid.length && vid.fps ? (vid.length / vid.fps).toFixed(2) : '?'} s)</span>] as [string, React.ReactNode]]
            : [['stills', `${snap.stills?.variants ?? 1} variant(s) per scene`] as [string, React.ReactNode]]),
    ['face swap', snap.face_swap?.enabled ? <span key="f">{wf(snap.face_swap.workflow)} · {snap.face_swap.restore_model}</span> : 'off'],
    ['evaluate', snap.evaluate?.enabled ? `on · up to ${snap.evaluate.max_attempts} attempts · review below ${snap.evaluate.review_threshold}/10` : 'off'],
    ['edit workflow', wf(snap.edit_workflow)],
    ['post-steps', snap.post_steps?.length ? <span key="ps">{snap.post_steps.map((s, i) => <span key={i} className="mr-2">{wf(s.workflow)}</span>)}</span> : 'none'],
    ['style', [snap.style?.keyframe_prefix && `prefix “${snap.style.keyframe_prefix}”`, snap.style?.animation_suffix && `motion “${snap.style.animation_suffix}”`, snap.style?.keyframe_negative && `negative “${snap.style.keyframe_negative}”`].filter(Boolean).join(' · ') || '—'],
  ]
  return (
    <dl className="grid gap-x-4 gap-y-1 text-sm sm:grid-cols-[9rem_1fr]">
      {rows.map(([k, v]) => (
        <div key={k} className="contents"><dt className="text-neutral-500">{k}</dt><dd className="break-words">{v}</dd></div>
      ))}
    </dl>
  )
}
