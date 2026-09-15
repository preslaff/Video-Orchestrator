'use client'

import { useState } from 'react'
import { errorMessage } from '@/api/client'
import type { AssetOut, DecisionOut, JobPublic, SceneEdit } from '@/api/types'
import { AssetPicker } from '@/components/AssetPicker'
import { useApprove, useRegenerateScene, useUpdateScenes } from '@/hooks/useJobs'
import { useUploadAsset } from '@/hooks/useSetup'
import { Button, ErrorBox } from '@/components/ui'

type Draft = { description: string; keyframe_prompt: string; keyframe_negative: string; animation_prompt: string; animation_negative: string; mode: 't2i' | 'edit'; references: string; reference_asset_ids: string[]; identity_asset_id: string | null; chosen_variant: number | null }

type PickerKind = 'refs' | 'identity' | 'image'

/** A one-line summary of chosen assets (small thumbnails + labels) for a collapsed picker. */
function PickSummary({ assets, ids, empty }: { assets: AssetOut[]; ids: string[]; empty: string }) {
  const chosen = ids.map((id) => assets.find((a) => a.id === id)).filter((a): a is AssetOut => !!a)
  if (!chosen.length) return <span className="text-xs text-neutral-500">{empty}</span>
  return (
    <span className="flex min-w-0 flex-wrap items-center gap-1">
      {chosen.map((a, i) => (
        <span key={a.id} className="flex max-w-48 items-center gap-1 rounded bg-neutral-100 px-1 py-0.5 text-xs" title={a.note_caption || a.label || a.name}>
          {chosen.length > 1 && <span className="text-neutral-500">{i + 1}.</span>}
          {/* eslint-disable-next-line @next/next/no-img-element -- served by FastAPI */}
          <img src={a.thumb_url || a.url} alt="" loading="lazy" className="h-6 w-6 rounded object-cover" />
          <span className="truncate">{a.label || a.name}</span>
        </span>
      ))}
    </span>
  )
}

function draftsFrom(job: JobPublic): Record<number, Draft> {
  const out: Record<number, Draft> = {}
  for (const s of job.scenes) {
    const chosen = s.variants.findIndex((v) => v.path === s.keyframe_path)
    out[s.index] = {
      description: s.description, keyframe_prompt: s.keyframe_prompt, keyframe_negative: s.keyframe_negative,
      animation_prompt: s.animation_prompt, animation_negative: s.animation_negative,
      mode: s.mode as 't2i' | 'edit', references: s.references.join('\n'), reference_asset_ids: [...(s.reference_asset_ids ?? [])],
      identity_asset_id: s.identity_asset_id ?? null, chosen_variant: chosen >= 0 ? chosen : null,
    }
  }
  return out
}

/**
 * Review + edit whenever the job is not running: at an approval gate, after it is done,
 * failed or stopped. Prompts, mode and references are always editable — on a scene whose
 * keyframe exists they apply on "Regenerate keyframe" / "Re-animate" — the variant picker
 * chooses which candidate is the scene's keyframe, "Use my image" replaces the keyframe
 * with a project image (or an upload), and Approve continues the run.
 */
export function GatePanel({ job, decisions = [], onStarted, assets = [] }: { job: JobPublic; decisions?: DecisionOut[]; onStarted?: () => void; assets?: AssetOut[] }) {
  const [drafts, setDrafts] = useState(() => draftsFrom(job))
  // At most one picker is open across the whole panel: a 40-scene job renders one full picker, not 120.
  const [openPicker, setOpenPicker] = useState<{ scene: number; kind: PickerKind } | null>(null)
  const isOpen = (scene: number, kind: PickerKind) => openPicker?.scene === scene && openPicker.kind === kind
  const togglePicker = (scene: number, kind: PickerKind) => setOpenPicker(isOpen(scene, kind) ? null : { scene, kind })
  const update = useUpdateScenes(job.id)
  const approve = useApprove(job.id)
  const regen = useRegenerateScene(job.id)
  const upload = useUploadAsset(job.project_id ?? '')
  const orphaned = job.status === 'running' && !job.active
  const editable = job.status === 'needs_approval' || job.status === 'done' || job.status === 'failed' || job.status === 'cancelled' || orphaned
  if (!editable) return null
  const video = job.kind === 'video'
  const busy = update.isPending || approve.isPending || regen.isPending || upload.isPending
  const slots = job.edit_reference_slots ?? 0
  const faces = assets.filter((a) => a.kind === 'master_face')
  const refAssets = assets.filter((a) => a.kind !== 'master_face')
  /** identity edits (asset id, '' = none) only when the project has faces; else leave untouched */
  const identityEdit = (d: Draft): string | null => (faces.length ? (d.identity_asset_id ?? '') : null)
  // A job in a project picks references as assets (and can upload new ones); otherwise as file paths.
  const assetRefs = refAssets.length > 0 || !!job.project_id
  const refsEdit = (d: Draft): string[] | null => (assetRefs ? d.reference_asset_ids : null)

  const regenerate = (index: number, what: 'keyframe' | 'video') => {
    const d = drafts[index]
    const cost = what === 'keyframe'
      ? `Regenerate the keyframe of scene ${index + 1}? Its face swap and clip are discarded and the keyframes gate reopens for review.`
      : `Re-animate scene ${index + 1} with Wan (several minutes on this GPU)?`
    if (!confirm(cost)) return
    regen.mutate({
      index,
      body: {
        what,
        description: d.description,
        keyframe_prompt: d.keyframe_prompt,
        keyframe_negative: d.keyframe_negative,
        animation_prompt: video ? d.animation_prompt : null,
        animation_negative: video ? d.animation_negative : null,
        mode: what === 'keyframe' ? d.mode : null,
        references: what === 'keyframe' && !assetRefs ? d.references.split('\n').map((x) => x.trim()).filter(Boolean) : null,
        reference_asset_ids: what === 'keyframe' ? refsEdit(d) : null,
        identity_asset_id: job.face_swap ? identityEdit(d) : null,   // a changed face applies to the redone swap
        apply_suggestion: false, // suggestions are applied in the editable prompt above
      },
    }, { onSuccess: () => onStarted?.() })
  }

  const set = (i: number, patch: Partial<Draft>) => setDrafts((d) => ({ ...d, [i]: { ...d[i], ...patch } }))

  /** Upload a file as a project reference asset, then hand its id on. */
  const uploadThen = (file: File, then: (assetId: string) => void) =>
    upload.mutate({ file, kind: 'reference' }, { onSuccess: (a: AssetOut) => then(a.id) })

  const chooseImage = (index: number, assetId: string) => {
    const scene = job.scenes.find((x) => x.index === index)
    const name = assets.find((a) => a.id === assetId)
    const label = name ? (name.label || name.name) : 'the uploaded image'
    const redo = video
      ? 'Its face swap and clip are discarded and the keyframes gate reopens for review.'
      : 'Its face swap and post-steps are redone.'
    const swapNote = job.face_swap && scene?.identity
      ? '\n\nThe scene\'s face will be swapped onto the image — choose identity "none" first to keep it as is.'
      : ''
    if (!confirm(`Use “${label}” as the keyframe of scene ${index + 1}? ${redo}${swapNote}`)) return
    setOpenPicker(null)
    regen.mutate({ index, body: { what: 'keyframe', keyframe_asset_id: assetId, apply_suggestion: false } },
      { onSuccess: () => onStarted?.() })
  }

  /** Evaluator fix suggestions for a scene: every logged retry + the stored verdict, newest first. */
  const suggestionsFor = (index: number): string[] => {
    const scene = job.scenes.find((x) => x.index === index)
    const out: string[] = []
    const fromVerdict = scene?.keyframe_verdict?.fix_suggestion
    if (fromVerdict) out.push(fromVerdict)
    for (const d of [...decisions].reverse()) {
      if (d.scene_index === index && d.phase === 'keyframe' && d.action === 'retry' && d.reason && !out.includes(d.reason)) out.push(d.reason)
    }
    return out
  }

  const applySuggestion = (index: number, text: string) => {
    const d = drafts[index]
    if (d.keyframe_prompt.includes(text)) return
    set(index, { keyframe_prompt: `${d.keyframe_prompt.replace(/[.\s]+$/, '')}. Correction: ${text}` })
  }

  const save = () => {
    const scenes: SceneEdit[] = job.scenes.map((s) => {
      const d = drafts[s.index]
      return {
        index: s.index,
        description: d.description,
        keyframe_prompt: d.keyframe_prompt,
        keyframe_negative: d.keyframe_negative,
        animation_prompt: video ? d.animation_prompt : null,
        animation_negative: video ? d.animation_negative : null,
        mode: s.keyframe_path ? null : d.mode,
        references: s.keyframe_path || assetRefs ? null : d.references.split('\n').map((x) => x.trim()).filter(Boolean),
        reference_asset_ids: s.keyframe_path ? null : refsEdit(d),
        identity_asset_id: job.face_swap && !s.video_path ? identityEdit(d) : null,
        chosen_variant: d.chosen_variant,
      }
    })
    update.mutate(scenes)
  }

  return (
    <div className="space-y-3 rounded-lg border border-amber-200 bg-amber-50 p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h2 className="font-semibold">{job.pending_gate ? `Gate: ${job.pending_gate}` : 'Post-run edits'}</h2>
          <p className="text-sm text-neutral-700">
            {job.pending_gate === 'storyboard' && 'Review the storyboard and prompts, then approve to generate keyframes.'}
            {job.pending_gate === 'keyframes' && 'Review the keyframes: regenerate any scene (optionally with an edited prompt), pick a variant, tweak animation prompts, then approve to render the clips (the expensive Wan phase).'}
            {!job.pending_gate && 'Regenerate a keyframe or re-animate a scene (the export folder is rebuilt when the job completes again), or pick a different variant and re-export.'}
          </p>
        </div>
        <div className="flex gap-2">
          <Button onClick={save} disabled={update.isPending}>{update.isPending ? 'Saving…' : 'Save changes'}</Button>
          {job.pending_gate && (
            <Button variant="primary" disabled={approve.isPending} onClick={() => approve.mutate(job.pending_gate!, { onSuccess: () => onStarted?.() })}>
              {approve.isPending ? 'Approving…' : `Approve ${job.pending_gate}`}
            </Button>
          )}
        </div>
      </div>
      <ErrorBox message={update.error ? errorMessage(update.error) : approve.error ? errorMessage(approve.error) : regen.error ? errorMessage(regen.error) : upload.error ? errorMessage(upload.error) : null} />
      {update.isSuccess && !update.isPending && <p className="text-xs text-green-700">Saved.</p>}

      <div className="grid gap-3 md:grid-cols-2">
        {job.scenes.map((s) => {
          const d = drafts[s.index]
          const locked = !!s.keyframe_path
          return (
            <div key={s.index} className="space-y-2 rounded border border-neutral-200 bg-white p-3 text-sm">
              <div className="flex items-center justify-between gap-2">
                <div className="font-medium">Scene {s.index + 1} <span className="text-xs font-normal text-neutral-500">{locked ? 'keyframe made — prompt edits apply on regenerate' : 'prompts editable'}</span></div>
                {locked && (
                  <div className="flex gap-1">
                    <Button disabled={busy} onClick={() => regenerate(s.index, 'keyframe')} title="new keyframe (new seed, current prompts); swap + clip are redone after approval">Regenerate keyframe</Button>
                    {video && <Button disabled={busy} onClick={() => regenerate(s.index, 'video')} title={s.video_path ? 'keep the keyframe, render the clip again with the current animation prompt' : 'keep the keyframe, render the (missing) clip with the current animation prompt'}>{s.video_path ? 'Re-animate' : 'Animate'}</Button>}
                  </div>
                )}
              </div>
              <label className="block"><span className="text-xs text-neutral-500">description</span>
                <textarea value={d.description} onChange={(e) => set(s.index, { description: e.target.value })} rows={2} className="mt-0.5 block w-full rounded border border-neutral-300 px-2 py-1" /></label>
              <label className="block"><span className="text-xs text-neutral-500">keyframe prompt</span>
                <textarea value={d.keyframe_prompt} onChange={(e) => set(s.index, { keyframe_prompt: e.target.value })} rows={3} className="mt-0.5 block w-full rounded border border-neutral-300 px-2 py-1" /></label>
              <label className="block"><span className="text-xs text-neutral-500">keyframe negative</span>
                <input value={d.keyframe_negative} onChange={(e) => set(s.index, { keyframe_negative: e.target.value })} className="mt-0.5 block w-full rounded border border-neutral-300 px-2 py-1" /></label>
              {suggestionsFor(s.index).length > 0 && (
                <div className="space-y-1 rounded bg-amber-50 p-2 text-xs text-amber-900">
                  <div className="font-semibold">Evaluator suggestions{s.keyframe_verdict?.decision === 'retry' ? ' (last verdict: retry)' : ''}</div>
                  {suggestionsFor(s.index).map((text, i) => {
                    const applied = d.keyframe_prompt.includes(text)
                    return (
                      <div key={i} className="flex items-start gap-2">
                        <p className="flex-1">{text}</p>
                        <button type="button" disabled={applied} onClick={() => applySuggestion(s.index, text)}
                          className="shrink-0 rounded border border-amber-300 bg-white px-2 py-0.5 font-medium disabled:opacity-50">
                          {applied ? 'applied' : 'Apply to prompt'}
                        </button>
                      </div>
                    )
                  })}
                  <p className="text-amber-700">Applying appends “Correction: …” to the keyframe prompt, exactly like the automatic retry; edit it, then Regenerate keyframe.</p>
                </div>
              )}
              {job.project_id && (
                <div>
                  <button type="button" disabled={busy} className="text-xs text-neutral-700 underline-offset-2 hover:underline disabled:opacity-50"
                    data-testid={`use-image-${s.index}`} onClick={() => togglePicker(s.index, 'image')}>
                    {isOpen(s.index, 'image') ? 'Cancel' : 'Use my image as keyframe…'}
                  </button>
                  {isOpen(s.index, 'image') && (
                    <div className="mt-1">
                      <span className="text-xs text-neutral-500">pick a project image or upload one — it replaces the keyframe (not generated, not evaluated)</span>
                      <AssetPicker assets={assets} selected={[]} disabled={busy} testId={`keyframe-image-${s.index}`}
                        onChange={(ids) => ids[0] && chooseImage(s.index, ids[0])}
                        onUpload={(f) => uploadThen(f, (id) => chooseImage(s.index, id))} uploading={upload.isPending} />
                    </div>
                  )}
                </div>
              )}
              <div className="grid grid-cols-[8rem_1fr] gap-2">
                {locked && <span className="col-span-2 text-xs text-neutral-500">mode &amp; references apply on Regenerate keyframe</span>}
                <label className="block"><span className="text-xs text-neutral-500">mode</span>
                  <select value={d.mode} onChange={(e) => set(s.index, { mode: e.target.value as 't2i' | 'edit' })} className="mt-0.5 block w-full rounded border border-neutral-300 bg-white px-2 py-1"><option value="t2i">t2i</option><option value="edit">edit</option></select></label>
                {d.mode === 'edit' && !assetRefs && (
                  <label className="block"><span className="text-xs text-neutral-500">reference paths (one per line)</span>
                    <textarea value={d.references} onChange={(e) => set(s.index, { references: e.target.value })} rows={2} className="mt-0.5 block w-full rounded border border-neutral-300 px-2 py-1 font-mono text-xs" /></label>
                )}
                {d.mode === 'edit' && assetRefs && (
                  <div className="min-w-0"><span className="text-xs text-neutral-500">references (project assets, in order){slots > 0 ? ` — the first ${slots} are used` : ''}</span>
                    <div className="flex items-center gap-2">
                      <PickSummary assets={assets} ids={d.reference_asset_ids} empty="no references" />
                      <button type="button" className="shrink-0 text-xs text-neutral-700 hover:underline" data-testid={`refs-${s.index}-choose`}
                        onClick={() => togglePicker(s.index, 'refs')}>{isOpen(s.index, 'refs') ? 'Done' : 'Choose…'}</button>
                    </div>
                    {isOpen(s.index, 'refs') && (
                      <AssetPicker assets={refAssets} selected={d.reference_asset_ids} multi onChange={(ids) => set(s.index, { reference_asset_ids: ids })} testId={`refs-${s.index}`}
                        onUpload={job.project_id ? (f) => uploadThen(f, (id) => setDrafts((all) => ({
                          ...all, [s.index]: { ...all[s.index], reference_asset_ids: [...all[s.index].reference_asset_ids, id] },
                        }))) : undefined}
                        uploading={upload.isPending} />
                    )}
                    {slots > 0 && d.reference_asset_ids.length > slots && (
                      <p className="text-xs text-amber-700">Only the first {slots} are used by the edit workflow; the rest are ignored.</p>
                    )}</div>
                )}
              </div>
              {job.face_swap && faces.length > 0 && (
                <div>
                  <span className="text-xs text-neutral-500">identity for the face swap{s.video_path ? ' (applies on Re-animate)' : s.swapped_path ? ' (applies on Animate)' : ''}</span>
                  <div className="flex items-center gap-2">
                    <PickSummary assets={faces} ids={d.identity_asset_id ? [d.identity_asset_id] : []} empty="none (no swap)" />
                    <button type="button" className="shrink-0 text-xs text-neutral-700 hover:underline" data-testid={`identity-${s.index}-choose`}
                      onClick={() => togglePicker(s.index, 'identity')}>{isOpen(s.index, 'identity') ? 'Done' : 'Choose…'}</button>
                  </div>
                  {isOpen(s.index, 'identity') && (
                    <AssetPicker assets={faces} selected={d.identity_asset_id ? [d.identity_asset_id] : []} none="none" testId={`identity-${s.index}`}
                      onChange={(ids) => set(s.index, { identity_asset_id: ids[0] ?? null })} />
                  )}
                </div>
              )}
              {video && (
                <>
                  <label className="block"><span className="text-xs text-neutral-500">animation prompt{s.video_path ? ' (edits apply on re-animate)' : ''}</span>
                    <textarea value={d.animation_prompt} onChange={(e) => set(s.index, { animation_prompt: e.target.value })} rows={2} className="mt-0.5 block w-full rounded border border-neutral-300 px-2 py-1" /></label>
                  <label className="block"><span className="text-xs text-neutral-500">animation negative</span>
                    <input value={d.animation_negative} onChange={(e) => set(s.index, { animation_negative: e.target.value })} className="mt-0.5 block w-full rounded border border-neutral-300 px-2 py-1" /></label>
                </>
              )}
              {s.variants.length > 1 && (
                <fieldset className="flex flex-wrap gap-2">
                  <legend className="text-xs text-neutral-500">keyframe variant</legend>
                  {s.variants.map((v, i) => (
                    <label key={i} className="flex items-center gap-1 rounded border border-neutral-300 px-2 py-1">
                      <input type="radio" name={`variant-${s.index}`} checked={d.chosen_variant === i} onChange={() => set(s.index, { chosen_variant: i })} />
                      {v.source === 'user' ? 'your image' : `v${i + 1}`}{v.verdict ? ` · ${v.verdict.score}/10` : ''}
                    </label>
                  ))}
                </fieldset>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
