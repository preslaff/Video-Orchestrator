'use client'

import { useMemo, useState } from 'react'
import { useRouter } from 'next/navigation'
import { errorDetails, errorMessage } from '@/api/client'
import type { JobSpec, OutputProfile } from '@/api/types'
import { useCreateJob } from '@/hooks/useJobs'
import { useEnv, useProfiles, useProject, useProjects, type EnvReport } from '@/hooks/useSetup'
import { EnvSummary } from '@/components/EnvSummary'
import { Button, Card, ErrorBox } from '@/components/ui'
import { clipSeconds } from '@/lib/wan'

const STEPS = ['Project', 'Profile', 'Brief', 'Review'] as const

export function profileSummary(p: OutputProfile): string {
  const kf = `${p.keyframe?.width ?? '?'}×${p.keyframe?.height ?? '?'}`
  if (p.kind === 'stills') return `stills · ${p.aspect} · ${kf} · ${p.stills?.variants ?? 1} variant(s)`
  const v = p.video
  if (!v) return `video · ${p.aspect} · ${kf}`
  return `video · ${p.aspect} · ${kf} → ${v.width}×${v.height} · ${v.length} frames @ ${v.fps} fps (~${clipSeconds(v.length, v.fps)}s)`
}

export function NewJobWizard({ initialProjectId = '' }: { initialProjectId?: string } = {}) {
  const router = useRouter()
  const [step, setStep] = useState(initialProjectId ? 1 : 0)  // a preselected project skips straight to the profile step
  const [projectId, setProjectId] = useState<string>(initialProjectId)
  const [profileId, setProfileId] = useState<string>('')
  const [idea, setIdea] = useState('')
  const [style, setStyle] = useState('')
  const [nScenes, setNScenes] = useState<number | ''>('')
  const [seed, setSeed] = useState<number>(0)
  const [autonomy, setAutonomy] = useState<'semi' | 'full'>('semi')
  const [mode, setMode] = useState<'t2i' | 'edit'>('t2i')
  const [refIds, setRefIds] = useState<string[]>([])
  const [masterFace, setMasterFace] = useState('')
  const [faceSwap, setFaceSwap] = useState<'profile' | 'on' | 'off'>('profile')
  const [evaluate, setEvaluate] = useState<'profile' | 'on' | 'off'>('profile')
  const [force, setForce] = useState(false)

  const projects = useProjects()
  const project = useProject(projectId || undefined)
  const profiles = useProfiles()
  const effectiveProfileId = profileId || project.data?.default_profile_id || 'cinematic_16x9'
  const profile = profiles.data?.find((p) => p.id === effectiveProfileId)
  const env = useEnv(effectiveProfileId, false, step === 3)
  const create = useCreateJob()

  const masterAsset = project.data?.assets.find((a) => a.kind === 'master_face')
  const swapOn = faceSwap === 'on' || (faceSwap === 'profile' && !!profile?.face_swap?.enabled)
  const refAssets = project.data?.assets.filter((a) => a.kind !== 'master_face') ?? []

  const problems = useMemo(() => {
    const out: string[] = []
    if (!idea.trim()) out.push('An idea is required.')
    if (swapOn && !masterFace && !masterAsset) out.push('Face swap is on but there is no master face (add one to the project or paste a path).')
    if (mode === 'edit' && refIds.length === 0) out.push('Edit mode needs at least one reference asset.')
    return out
  }, [idea, swapOn, masterFace, masterAsset, mode, refIds])

  const envFailures = env.data?.checks.filter((c) => c.status === 'fail') ?? []
  const blocked = problems.length > 0 || (envFailures.length > 0 && !force)
  const submitError = create.error ? errorMessage(create.error) : null
  const submitReport = create.error ? (errorDetails(create.error) as EnvReport | undefined) : undefined

  const spec: JobSpec = {
    idea: idea.trim(),
    style,
    autonomy,
    n_scenes: nScenes === '' ? null : Number(nScenes),
    seed,
    project_id: projectId || null,
    profile_id: profileId || null,
    mode,
    reference_asset_ids: mode === 'edit' ? refIds : [],
    reference_images: [],
    master_face: masterFace,
    face_swap: faceSwap === 'profile' ? null : faceSwap === 'on',
    evaluate: evaluate === 'profile' ? null : evaluate === 'on',
    force,
  }

  const submit = () => {
    create.mutate(spec, { onSuccess: (job) => router.push(`/jobs/${job.id}`) })
  }

  return (
    <div className="space-y-4">
      <ol className="flex gap-2 text-sm">
        {STEPS.map((s, i) => (
          <li key={s}>
            <button type="button" onClick={() => i < step && setStep(i)} className={`rounded px-3 py-1 ${i === step ? 'bg-neutral-900 text-white' : i < step ? 'bg-neutral-200' : 'bg-neutral-100 text-neutral-400'}`}>
              {i + 1}. {s}
            </button>
          </li>
        ))}
      </ol>

      {step === 0 && (
        <Card title="Project (optional)">
          <label className="block text-sm">
            <span className="text-neutral-600">Group this job under a project to reuse its master face, references and export folder.</span>
            <select data-testid="project" value={projectId} onChange={(e) => { setProjectId(e.target.value); setProfileId(''); setRefIds([]) }} className="mt-2 block w-full rounded border border-neutral-300 bg-white px-2 py-1">
              <option value="">— no project —</option>
              {projects.data?.map((p) => <option key={p.id} value={p.id}>{p.name} ({p.n_assets} asset{p.n_assets === 1 ? '' : 's'})</option>)}
            </select>
          </label>
          {project.data && (
            <div className="mt-3 flex items-start gap-4">
              {masterAsset ? (
                // eslint-disable-next-line @next/next/no-img-element -- served by FastAPI
                <img src={masterAsset.url} alt={masterAsset.label || masterAsset.name} className="h-24 w-24 rounded object-cover ring-1 ring-neutral-200" />
              ) : (
                <div className="flex h-24 w-24 items-center justify-center rounded bg-neutral-100 text-center text-xs text-neutral-400">no master face</div>
              )}
              <div className="text-sm text-neutral-600">
                <p>default profile <b>{project.data.default_profile_id}</b></p>
                <p>master face {masterAsset ? <b>{masterAsset.label || masterAsset.name}</b> : <span className="text-amber-700">none — add one under Projects, or paste a path in step 3</span>}</p>
                <p>{refAssets.length} reference/product asset(s) for edit mode</p>
                {project.data.assets.filter((a) => a.kind === 'master_face').length > 1 && <p className="text-xs text-sky-800">{project.data.assets.filter((a) => a.kind === 'master_face').length} faces: the storyboard assigns one per scene from their labels (or none); adjust at the storyboard gate.</p>}
                <p>export → <code className="text-xs">{project.data.export_root_effective}</code></p>
              </div>
            </div>
          )}
        </Card>
      )}

      {step === 1 && (
        <Card title="Output profile">
          <div className="space-y-2">
            {profiles.data?.map((p) => (
              <label key={p.id} className={`flex cursor-pointer items-start gap-3 rounded border p-2 ${effectiveProfileId === p.id ? 'border-neutral-900 bg-neutral-50' : 'border-neutral-200'}`}>
                <input type="radio" name="profile" value={p.id} checked={effectiveProfileId === p.id} onChange={() => setProfileId(p.id)} className="mt-1" />
                <span>
                  <span className="font-medium">{p.name}</span> <span className="text-xs text-neutral-500">{p.id}{p.builtin ? ' · built-in' : ''}</span>
                  <br />
                  <span className="text-sm text-neutral-600">{profileSummary(p)} · {p.purpose}</span>
                </span>
              </label>
            ))}
          </div>
        </Card>
      )}

      {step === 2 && (
        <Card title="Brief">
          <div className="grid gap-3 md:grid-cols-2">
            <label className="block text-sm md:col-span-2">
              <span className="font-medium">Idea</span>
              <textarea data-testid="idea" value={idea} onChange={(e) => setIdea(e.target.value)} rows={3} className="mt-1 block w-full rounded border border-neutral-300 px-2 py-1" placeholder="A woman kneads bread at dawn in a stone kitchen…" />
            </label>
            <label className="block text-sm">
              <span className="font-medium">Style (free text, goes first in every keyframe prompt)</span>
              <input value={style} onChange={(e) => setStyle(e.target.value)} className="mt-1 block w-full rounded border border-neutral-300 px-2 py-1" placeholder="impasto oil painting / 35mm film still / phone photo" />
            </label>
            <div className="grid grid-cols-3 gap-2">
              <label className="block text-sm"><span className="font-medium">Scenes</span><input type="number" min={1} max={12} value={nScenes} onChange={(e) => setNScenes(e.target.value === '' ? '' : Number(e.target.value))} className="mt-1 block w-full rounded border border-neutral-300 px-2 py-1" placeholder="auto" /></label>
              <label className="block text-sm"><span className="font-medium">Seed</span><input type="number" min={0} value={seed} onChange={(e) => setSeed(Number(e.target.value) || 0)} className="mt-1 block w-full rounded border border-neutral-300 px-2 py-1" /><span className="text-xs text-neutral-500">0 = random</span></label>
              <label className="block text-sm"><span className="font-medium">Autonomy</span><select value={autonomy} onChange={(e) => setAutonomy(e.target.value as 'semi' | 'full')} className="mt-1 block w-full rounded border border-neutral-300 bg-white px-2 py-1"><option value="semi">semi (gates)</option><option value="full">full</option></select></label>
            </div>
            <label className="block text-sm">
              <span className="font-medium">Keyframe mode</span>
              <select data-testid="mode" value={mode} onChange={(e) => setMode(e.target.value as 't2i' | 'edit')} className="mt-1 block w-full rounded border border-neutral-300 bg-white px-2 py-1">
                <option value="t2i">text-to-image (new scenes)</option>
                <option value="edit">reference-locked edit (same composition, light edit)</option>
              </select>
            </label>
            {mode === 'edit' && (
              <div className="text-sm md:col-span-2">
                <span className="font-medium">References (project assets)</span>
                <span className="ml-2 text-xs text-neutral-500">{refIds.length} selected · order = click order; the first is the primary reference</span>
                {!projectId && <p className="text-amber-700">Pick a project in step 1 to use its reference/product assets.</p>}
                {projectId && refAssets.length === 0 && <p className="text-amber-700">The selected project has no reference/product assets — add some under Projects.</p>}
                {refAssets.length > 0 && (
                  <div className="mt-2 grid grid-cols-3 gap-2 sm:grid-cols-4 lg:grid-cols-6">
                    {refAssets.map((a) => {
                      const pos = refIds.indexOf(a.id)
                      const selected = pos >= 0
                      return (
                        <button
                          key={a.id}
                          type="button"
                          data-testid={`ref-${a.id}`}
                          aria-pressed={selected}
                          onClick={() => setRefIds(selected ? refIds.filter((x) => x !== a.id) : [...refIds, a.id])}
                          className={`group relative overflow-hidden rounded border text-left ${selected ? 'border-neutral-900 ring-2 ring-neutral-900' : 'border-neutral-200 hover:border-neutral-400'}`}
                          title={a.label || a.name}
                        >
                          {/* eslint-disable-next-line @next/next/no-img-element -- served by FastAPI */}
                          <img src={a.url} alt={a.label || a.name} className="aspect-square w-full bg-neutral-100 object-cover" />
                          {selected && <span className="absolute left-1 top-1 rounded-full bg-neutral-900 px-1.5 text-xs font-semibold text-white">{pos + 1}</span>}
                          <span className="block truncate px-1 py-0.5 text-xs text-neutral-700">
                            <span className="rounded bg-neutral-100 px-1">{a.kind}</span> {a.label || a.name}
                          </span>
                        </button>
                      )
                    })}
                  </div>
                )}
              </div>
            )}
            <label className="block text-sm"><span className="font-medium">Face swap</span><select data-testid="face_swap" value={faceSwap} onChange={(e) => setFaceSwap(e.target.value as typeof faceSwap)} className="mt-1 block w-full rounded border border-neutral-300 bg-white px-2 py-1"><option value="profile">profile default ({profile?.face_swap?.enabled ? 'on' : 'off'})</option><option value="on">on</option><option value="off">off</option></select></label>
            <label className="block text-sm"><span className="font-medium">Evaluate keyframes (Gemma vision)</span><select value={evaluate} onChange={(e) => setEvaluate(e.target.value as typeof evaluate)} className="mt-1 block w-full rounded border border-neutral-300 bg-white px-2 py-1"><option value="profile">profile default ({profile?.evaluate?.enabled ? 'on' : 'off'})</option><option value="on">on</option><option value="off">off</option></select></label>
            {swapOn && (
              <label className="block text-sm md:col-span-2">
                <span className="font-medium">Master face path</span>
                <input data-testid="master_face" value={masterFace} onChange={(e) => setMasterFace(e.target.value)} className="mt-1 block w-full rounded border border-neutral-300 px-2 py-1 font-mono text-xs" placeholder={masterAsset ? `project asset: ${masterAsset.name}` : '/path/to/face.jpg'} />
              </label>
            )}
          </div>
        </Card>
      )}

      {step === 3 && (
        <Card title="Review & submit">
          <dl className="grid gap-x-6 gap-y-1 text-sm md:grid-cols-[10rem_1fr]">
            <dt className="text-neutral-500">idea</dt><dd>{idea || <em className="text-red-700">missing</em>}</dd>
            <dt className="text-neutral-500">project</dt><dd>{project.data?.name ?? '—'}</dd>
            <dt className="text-neutral-500">profile</dt><dd>{profile ? `${profile.name} — ${profileSummary(profile)}` : effectiveProfileId}</dd>
            <dt className="text-neutral-500">scenes / seed</dt><dd>{nScenes || 'auto'} / {seed || 'random'} · {autonomy}</dd>
            <dt className="text-neutral-500">mode</dt><dd>{mode}{mode === 'edit' ? ` (${refIds.length} ref)` : ''}</dd>
            <dt className="text-neutral-500">face swap</dt><dd>{swapOn ? `on (${masterFace || masterAsset?.name || 'no face!'})` : 'off'}</dd>
          </dl>
          <div className="mt-3 space-y-2">
            {problems.map((p) => <ErrorBox key={p} message={p} />)}
            <div className="rounded border border-neutral-200 p-2">
              <div className="mb-1 text-xs font-semibold uppercase text-neutral-500">Environment check {env.isFetching ? '(checking…)' : ''}</div>
              <EnvSummary report={env.data} />
              {envFailures.length > 0 && (
                <label className="mt-2 flex items-center gap-2 text-sm"><input data-testid="force" type="checkbox" checked={force} onChange={(e) => setForce(e.target.checked)} /> submit anyway (force)</label>
              )}
            </div>
            <ErrorBox message={submitError} />
            {submitReport?.checks && <EnvSummary report={submitReport} />}
          </div>
        </Card>
      )}

      <div className="flex justify-between">
        <Button disabled={step === 0} onClick={() => setStep(step - 1)}>Back</Button>
        {step < 3 ? (
          <Button variant="primary" onClick={() => setStep(step + 1)}>Next</Button>
        ) : (
          <Button data-testid="submit" variant="primary" disabled={blocked || create.isPending} onClick={submit}>{create.isPending ? 'Submitting…' : 'Start job'}</Button>
        )}
      </div>
    </div>
  )
}
