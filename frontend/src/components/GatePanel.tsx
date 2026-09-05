'use client'

import { useState } from 'react'
import { errorMessage } from '@/api/client'
import type { AssetOut, DecisionOut, JobPublic, SceneEdit } from '@/api/types'
import { AssetPicker } from '@/components/AssetPicker'
import { useApprove, useRegenerateScene, useUpdateScenes } from '@/hooks/useJobs'
import { Button, ErrorBox } from '@/components/ui'

type Draft = { description: string; keyframe_prompt: string; keyframe_negative: string; animation_prompt: string; animation_negative: string; mode: 't2i' | 'edit'; references: string; reference_asset_ids: string[]; identity_asset_id: string | null; chosen_variant: number | null }

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
 * Review + edit at an approval gate (and after the run): prompts are always
 * editable — edits to a scene whose artifact exists apply on "Regenerate
 * keyframe" / "Re-animate" — the variant picker chooses which candidate is the
 * scene's keyframe, and Approve continues the run.
 */
export function GatePanel({ job, decisions = [], onStarted, assets = [] }: { job: JobPublic; decisions?: DecisionOut[]; onStarted?: () => void; assets?: AssetOut[] }) {
  const [drafts, setDrafts] = useState(() => draftsFrom(job))
  const update = useUpdateScenes(job.id)
  const approve = useApprove(job.id)
  const regen = useRegenerateScene(job.id)
  const editable = job.status === 'needs_approval' || job.status === 'done' || job.status === 'failed'
  if (!editable) return null
  const video = job.kind === 'video'
  const busy = update.isPending || approve.isPending || regen.isPending
  const faces = assets.filter((a) => a.kind === 'master_face')
  const refAssets = assets.filter((a) => a.kind !== 'master_face')
  /** identity edits (asset id, '' = none) only when the project has faces; else leave untouched */
  const identityEdit = (d: Draft): string | null => (faces.length ? (d.identity_asset_id ?? '') : null)
  const refsEdit = (d: Draft): string[] | null => (refAssets.length ? d.reference_asset_ids : null)

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
        references: what === 'keyframe' && !refAssets.length ? d.references.split('\n').map((x) => x.trim()).filter(Boolean) : null,
        reference_asset_ids: what === 'keyframe' ? refsEdit(d) : null,
        identity_asset_id: job.face_swap ? identityEdit(d) : null,   // a changed face applies to the redone swap
        apply_suggestion: false, // suggestions are applied in the editable prompt above
      },
    }, { onSuccess: () => onStarted?.() })
  }

  const set = (i: number, patch: Partial<Draft>) => setDrafts((d) => ({ ...d, [i]: { ...d[i], ...patch } }))

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
        references: s.keyframe_path || refAssets.length ? null : d.references.split('\n').map((x) => x.trim()).filter(Boolean),
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
      <ErrorBox message={update.error ? errorMessage(update.error) : approve.error ? errorMessage(approve.error) : regen.error ? errorMessage(regen.error) : null} />
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
              {!locked && (
                <div className="grid grid-cols-[8rem_1fr] gap-2">
                  <label className="block"><span className="text-xs text-neutral-500">mode</span>
                    <select value={d.mode} onChange={(e) => set(s.index, { mode: e.target.value as 't2i' | 'edit' })} className="mt-0.5 block w-full rounded border border-neutral-300 bg-white px-2 py-1"><option value="t2i">t2i</option><option value="edit">edit</option></select></label>
                  {d.mode === 'edit' && !refAssets.length && (
                    <label className="block"><span className="text-xs text-neutral-500">reference paths (one per line)</span>
                      <textarea value={d.references} onChange={(e) => set(s.index, { references: e.target.value })} rows={2} className="mt-0.5 block w-full rounded border border-neutral-300 px-2 py-1 font-mono text-xs" /></label>
                  )}
                  {d.mode === 'edit' && refAssets.length > 0 && (
                    <div><span className="text-xs text-neutral-500">references (project assets, in order)</span>
                      <AssetPicker assets={refAssets} selected={d.reference_asset_ids} multi onChange={(ids) => set(s.index, { reference_asset_ids: ids })} testId={`refs-${s.index}`} /></div>
                  )}
                </div>
              )}
              {job.face_swap && faces.length > 0 && (
                <div>
                  <span className="text-xs text-neutral-500">identity for the face swap{s.video_path ? ' (applies on Re-animate)' : s.swapped_path ? ' (applies on Animate)' : ''}</span>
                  <AssetPicker assets={faces} selected={d.identity_asset_id ? [d.identity_asset_id] : []} none="none" testId={`identity-${s.index}`}
                    onChange={(ids) => set(s.index, { identity_asset_id: ids[0] ?? null })} />
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
                      v{i + 1}{v.verdict ? ` · ${v.verdict.score}/10` : ''}
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
