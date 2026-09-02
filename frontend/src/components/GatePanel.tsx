'use client'

import { useState } from 'react'
import { errorMessage } from '@/api/client'
import type { JobPublic, SceneEdit } from '@/api/types'
import { useApprove, useUpdateScenes } from '@/hooks/useJobs'
import { Button, ErrorBox } from '@/components/ui'

type Draft = { description: string; keyframe_prompt: string; keyframe_negative: string; animation_prompt: string; animation_negative: string; mode: 't2i' | 'edit'; references: string; chosen_variant: number | null }

function draftsFrom(job: JobPublic): Record<number, Draft> {
  const out: Record<number, Draft> = {}
  for (const s of job.scenes) {
    const chosen = s.variants.findIndex((v) => v.path === s.keyframe_path)
    out[s.index] = {
      description: s.description, keyframe_prompt: s.keyframe_prompt, keyframe_negative: s.keyframe_negative,
      animation_prompt: s.animation_prompt, animation_negative: s.animation_negative,
      mode: s.mode as 't2i' | 'edit', references: s.references.join('\n'), chosen_variant: chosen >= 0 ? chosen : null,
    }
  }
  return out
}

/**
 * Review + edit at an approval gate: prompts are editable until their artifact
 * exists (keyframe prompt locks once a keyframe was made), the variant picker
 * chooses which candidate is the scene's keyframe, and Approve continues the run.
 */
export function GatePanel({ job }: { job: JobPublic }) {
  const [drafts, setDrafts] = useState(() => draftsFrom(job))
  const update = useUpdateScenes(job.id)
  const approve = useApprove(job.id)
  const editable = job.status === 'needs_approval' || job.status === 'done'
  if (!editable) return null
  const video = job.kind === 'video'

  const set = (i: number, patch: Partial<Draft>) => setDrafts((d) => ({ ...d, [i]: { ...d[i], ...patch } }))

  const save = () => {
    const scenes: SceneEdit[] = job.scenes.map((s) => {
      const d = drafts[s.index]
      return {
        index: s.index,
        description: d.description,
        keyframe_prompt: s.keyframe_path ? null : d.keyframe_prompt,
        keyframe_negative: s.keyframe_path ? null : d.keyframe_negative,
        animation_prompt: video && !s.video_path ? d.animation_prompt : null,
        animation_negative: video && !s.video_path ? d.animation_negative : null,
        mode: s.keyframe_path ? null : d.mode,
        references: s.keyframe_path ? null : d.references.split('\n').map((x) => x.trim()).filter(Boolean),
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
            {job.pending_gate === 'keyframes' && 'Review the keyframes (pick a variant, tweak animation prompts), then approve to render the clips (the expensive Wan phase).'}
            {!job.pending_gate && 'Pick a different variant and re-export if needed.'}
          </p>
        </div>
        <div className="flex gap-2">
          <Button onClick={save} disabled={update.isPending}>{update.isPending ? 'Saving…' : 'Save changes'}</Button>
          {job.pending_gate && (
            <Button variant="primary" disabled={approve.isPending} onClick={() => approve.mutate(job.pending_gate!)}>
              {approve.isPending ? 'Approving…' : `Approve ${job.pending_gate}`}
            </Button>
          )}
        </div>
      </div>
      <ErrorBox message={update.error ? errorMessage(update.error) : approve.error ? errorMessage(approve.error) : null} />
      {update.isSuccess && !update.isPending && <p className="text-xs text-green-700">Saved.</p>}

      <div className="grid gap-3 md:grid-cols-2">
        {job.scenes.map((s) => {
          const d = drafts[s.index]
          const locked = !!s.keyframe_path
          return (
            <div key={s.index} className="space-y-2 rounded border border-neutral-200 bg-white p-3 text-sm">
              <div className="font-medium">Scene {s.index + 1} <span className="text-xs font-normal text-neutral-500">{locked ? 'keyframe made — prompt locked' : 'prompts editable'}</span></div>
              <label className="block"><span className="text-xs text-neutral-500">description</span>
                <textarea value={d.description} onChange={(e) => set(s.index, { description: e.target.value })} rows={2} className="mt-0.5 block w-full rounded border border-neutral-300 px-2 py-1" /></label>
              <label className="block"><span className="text-xs text-neutral-500">keyframe prompt</span>
                <textarea value={d.keyframe_prompt} readOnly={locked} onChange={(e) => set(s.index, { keyframe_prompt: e.target.value })} rows={3} className={`mt-0.5 block w-full rounded border border-neutral-300 px-2 py-1 ${locked ? 'bg-neutral-50 text-neutral-500' : ''}`} /></label>
              {!locked && (
                <label className="block"><span className="text-xs text-neutral-500">keyframe negative</span>
                  <input value={d.keyframe_negative} onChange={(e) => set(s.index, { keyframe_negative: e.target.value })} className="mt-0.5 block w-full rounded border border-neutral-300 px-2 py-1" /></label>
              )}
              {!locked && (
                <div className="grid grid-cols-[8rem_1fr] gap-2">
                  <label className="block"><span className="text-xs text-neutral-500">mode</span>
                    <select value={d.mode} onChange={(e) => set(s.index, { mode: e.target.value as 't2i' | 'edit' })} className="mt-0.5 block w-full rounded border border-neutral-300 bg-white px-2 py-1"><option value="t2i">t2i</option><option value="edit">edit</option></select></label>
                  {d.mode === 'edit' && (
                    <label className="block"><span className="text-xs text-neutral-500">reference paths (one per line)</span>
                      <textarea value={d.references} onChange={(e) => set(s.index, { references: e.target.value })} rows={2} className="mt-0.5 block w-full rounded border border-neutral-300 px-2 py-1 font-mono text-xs" /></label>
                  )}
                </div>
              )}
              {video && (
                <>
                  <label className="block"><span className="text-xs text-neutral-500">animation prompt</span>
                    <textarea value={d.animation_prompt} readOnly={!!s.video_path} onChange={(e) => set(s.index, { animation_prompt: e.target.value })} rows={2} className={`mt-0.5 block w-full rounded border border-neutral-300 px-2 py-1 ${s.video_path ? 'bg-neutral-50 text-neutral-500' : ''}`} /></label>
                  {!s.video_path && (
                    <label className="block"><span className="text-xs text-neutral-500">animation negative</span>
                      <input value={d.animation_negative} onChange={(e) => set(s.index, { animation_negative: e.target.value })} className="mt-0.5 block w-full rounded border border-neutral-300 px-2 py-1" /></label>
                  )}
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
