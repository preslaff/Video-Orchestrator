'use client'

import { useEffect, useRef, useState } from 'react'
import { errorMessage, isMissingRoute } from '@/api/client'
import type { AssetOut, DecisionOut, JobPublic, SceneEdit } from '@/api/types'
import { AssetPicker } from '@/components/AssetPicker'
import { Zoomable } from '@/components/Lightbox'
import { MediaThumb } from '@/components/MediaThumb'
import { LockRegions } from '@/components/LockRegions'
import { VersionStrip } from '@/components/VersionStrip'
import { profileSizes } from '@/components/SceneGrid'
import { isImageAsset, isVideoAsset } from '@/lib/assetFilter'
import { estimateClip, fmtRange } from '@/lib/render'
import { useApprove, useDeleteVersion, useRegenerateScene, useSaveSceneAsset, useSceneLocks, useSceneStructure, useRemoveNarration, useSpeakScene, useUpdateScenes } from '@/hooks/useJobs'
import { useUploadAsset, useWorkflows } from '@/hooks/useSetup'
import { Button, ErrorBox } from '@/components/ui'

type Draft = { description: string; keyframe_prompt: string; keyframe_negative: string; animation_prompt: string; animation_negative: string; mode: 't2i' | 'edit'; edit_workflow: string; edit_base: '' | 'keyframe'; references: string; reference_asset_ids: string[]; identity_asset_id: string | null; narration: string }

type PickerKind = 'refs' | 'identity' | 'image' | 'locks' | 'video'

/** Above this many scenes the rows start collapsed (a 3-scene job stays as it was). */
const AUTO_COLLAPSE_ABOVE = 6

/** Dispatched by the progress panel's scene chips; the gate panel opens that scene. */
export const GOTO_SCENE = 'vorch:goto-scene'

/** A one-line summary of chosen assets (small thumbnails + labels) for a collapsed picker. */
function PickSummary({ assets, ids, empty }: { assets: AssetOut[]; ids: string[]; empty: string }) {
  const chosen = ids.map((id) => assets.find((a) => a.id === id)).filter((a): a is AssetOut => !!a)
  if (!chosen.length) return <span className="text-xs text-neutral-500">{empty}</span>
  return (
    <span className="flex min-w-0 flex-wrap items-center gap-1">
      {chosen.map((a, i) => (
        <span key={a.id} className="flex max-w-56 items-center gap-1.5 rounded bg-neutral-100 px-1 py-0.5 text-xs" title={a.note_caption || a.label || a.name}>
          {chosen.length > 1 && <span className="text-neutral-500">{i + 1}.</span>}
          {/* Big enough to recognise the photo at a glance — this is what the scene is being
              generated from, and the label alone rarely tells two grey seas apart. */}
          <Zoomable items={[{ url: a.url, caption: a.label || a.name }]} className="shrink-0">
            {/* eslint-disable-next-line @next/next/no-img-element -- served by FastAPI */}
            <img src={a.thumb_url || a.url} alt="" loading="lazy" className="h-12 w-12 rounded object-cover" />
          </Zoomable>
          <span className="truncate">{a.label || a.name}</span>
        </span>
      ))}
    </span>
  )
}

/**
 * The storyboard's hint that one of the project's photos fits this scene. Nothing is applied until
 * "Use as reference", which only changes the draft (mode edit + that photo) — saved like any edit.
 * A "motif only" match says why t2i may still be better: edit mode copies the photo's framing.
 */
function ReferenceHint({ scene, asset, why, fit, inUse, onUse, onUndo }: {
  scene: number; asset: AssetOut; why: string; fit: string; inUse: boolean; onUse: () => void; onUndo: () => void
}) {
  const sameShot = fit === 'same_shot'
  return (
    <div className="flex items-start gap-2 rounded bg-sky-50 p-2 text-xs text-sky-950" data-testid={`hint-${scene}`}>
      <Zoomable items={[{ url: asset.url, caption: asset.label || asset.name }]} className="shrink-0" testId={`hint-${scene}-zoom`}>
        {/* eslint-disable-next-line @next/next/no-img-element -- served by FastAPI */}
        <img src={asset.thumb_url || asset.url} alt="" loading="lazy" className="h-12 w-12 rounded object-cover" />
      </Zoomable>
      <div className="min-w-0 flex-1">
        <div><span className="font-semibold">Suggested reference</span> · {sameShot ? 'same shot' : 'motif only'}</div>
        <div className="truncate" title={asset.note_caption || asset.label || asset.name}>{asset.label || asset.name}</div>
        {why && <div className="text-sky-800">{why}</div>}
        {!sameShot && !inUse && <div className="text-sky-700">Different framing — edit mode would keep the photo&apos;s composition; t2i may suit the scene better.</div>}
      </div>
      {inUse
        ? <button type="button" onClick={onUndo} data-testid={`hint-${scene}-undo`} className="shrink-0 rounded border border-sky-300 bg-white px-2 py-0.5 font-medium">In use · undo</button>
        : <button type="button" onClick={onUse} data-testid={`hint-${scene}-use`} className="shrink-0 rounded border border-sky-300 bg-white px-2 py-0.5 font-medium">Use as reference</button>}
    </div>
  )
}

/** The space between two scenes: the drop target for a dragged row, and where a new scene goes. */
function Gap({ at, active, onOver, onDrop, onAdd, adding }: {
  at: number; active: boolean; onOver: (at: number | null) => void; onDrop: (at: number) => void
  onAdd: () => void; adding: boolean
}) {
  return (
    <div
      onDragOver={(e) => { e.preventDefault(); onOver(at) }}
      onDragLeave={() => onOver(null)}
      onDrop={(e) => { e.preventDefault(); onDrop(at) }}
      data-testid={`scene-gap-${at}`}
      className={`group flex h-6 items-center justify-center ${active ? 'h-8' : ''}`}>
      {active ? (
        <span className="h-1 w-full rounded bg-blue-400" />
      ) : (
        <button type="button" onClick={onAdd} data-testid={`add-scene-${at}`}
          className="rounded px-2 py-0.5 text-xs text-neutral-400 opacity-0 transition hover:bg-neutral-100 hover:text-neutral-700 focus:opacity-100 group-hover:opacity-100">
          {adding ? 'adding…' : '+ add scene here'}
        </button>
      )}
    </div>
  )
}

/** "Add scene here": a description is all Gemma needs to write the scene's two prompts. */
function AddScene({ position, busy, onAdd, onCancel }: {
  position: number; busy: boolean; onAdd: (b: { description: string; setting: string; mood: string }) => void; onCancel: () => void
}) {
  const [description, setDescription] = useState('')
  const [setting, setSetting] = useState('')
  const [mood, setMood] = useState('')
  return (
    <div className="space-y-2 rounded-lg border-2 border-dashed border-neutral-300 bg-white p-3 text-sm" data-testid={`add-scene-form-${position}`}>
      <div className="font-medium">New scene at position {position + 1}</div>
      <label className="block"><span className="text-xs text-neutral-500">what happens in this scene</span>
        <textarea autoFocus value={description} onChange={(e) => setDescription(e.target.value)} rows={2}
          data-testid="add-scene-description" className="mt-0.5 block w-full rounded border border-neutral-300 px-2 py-1"
          placeholder="the tower's lamp turning at night, seen from the sea" /></label>
      <div className="grid gap-2 sm:grid-cols-2">
        <label className="block"><span className="text-xs text-neutral-500">setting (optional)</span>
          <input value={setting} onChange={(e) => setSetting(e.target.value)} data-testid="add-scene-setting"
            className="mt-0.5 block w-full rounded border border-neutral-300 px-2 py-1" /></label>
        <label className="block"><span className="text-xs text-neutral-500">mood (optional)</span>
          <input value={mood} onChange={(e) => setMood(e.target.value)} data-testid="add-scene-mood"
            className="mt-0.5 block w-full rounded border border-neutral-300 px-2 py-1" /></label>
      </div>
      <div className="flex items-center gap-2">
        <Button variant="primary" disabled={busy || !description.trim()} data-testid="add-scene-submit"
          onClick={() => onAdd({ description: description.trim(), setting: setting.trim(), mood: mood.trim() })}>
          {busy ? 'Adding…' : 'Add and write prompts'}
        </Button>
        <button type="button" className="text-xs text-neutral-600 hover:underline" onClick={onCancel}>Cancel</button>
        <span className="text-xs text-neutral-500">Gemma writes the keyframe and animation prompts (~40 s).</span>
      </div>
    </div>
  )
}

/** Drafts are keyed by the scene's **stable id**, never its index: a delete or a reorder
 *  renumbers positions, and keying by number would graft a half-typed prompt onto its neighbour. */
function draftsFrom(job: JobPublic): Record<string, Draft> {
  const out: Record<string, Draft> = {}
  for (const s of job.scenes) {
    out[s.id] = {
      description: s.description, keyframe_prompt: s.keyframe_prompt, keyframe_negative: s.keyframe_negative,
      animation_prompt: s.animation_prompt, animation_negative: s.animation_negative,
      narration: s.narration ?? '',
      mode: s.mode as 't2i' | 'edit', edit_workflow: s.edit_workflow ?? '', edit_base: (s.edit_base ?? '') as '' | 'keyframe',
      references: s.references.join('\n'), reference_asset_ids: [...(s.reference_asset_ids ?? [])],
      identity_asset_id: s.identity_asset_id ?? null,
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
  // What the server last told us each field was. A field the user has since changed is "dirty";
  // everything else follows the server, so regenerated media and other people's edits land here
  // without discarding a half-typed prompt (the page used to remount the whole panel instead).
  const serverDrafts = useRef(draftsFrom(job))
  // The same snapshot as state, because the dirty marker on a collapsed row is read while
  // rendering and a ref must not be (react-hooks/refs).
  const [saved, setSaved] = useState(() => draftsFrom(job))
  useEffect(() => {
    const next = draftsFrom(job)
    setDrafts((current) => {
      const merged: Record<string, Draft> = {}
      for (const s of job.scenes) {
        const mine = current[s.id], was = serverDrafts.current[s.id], now = next[s.id]
        if (!mine || !was) { merged[s.id] = now; continue }
        merged[s.id] = { ...now }
        for (const key of Object.keys(now) as (keyof Draft)[]) {
          const dirty = JSON.stringify(mine[key]) !== JSON.stringify(was[key])
          if (dirty) (merged[s.id] as Record<string, unknown>)[key] = mine[key]
        }
      }
      return merged
    })
    serverDrafts.current = next
    setSaved(next)
  }, [job])
  // Which scenes show their editor. A 40-scene job is unreadable fully expanded — reaching scene 33
  // meant scrolling past 32 editors — so above a handful every scene starts as a one-line row.
  const [open, setOpen] = useState<Set<string>>(
    () => new Set(job.scenes.length > AUTO_COLLAPSE_ABOVE ? [] : job.scenes.map((s) => s.id)),
  )
  const expanded = (id: string) => open.has(id)
  const toggleScene = (id: string) =>
    setOpen((cur) => {
      const next = new Set(cur)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  // The progress panel's scene chips jump to a row that may be collapsed: open it as well.
  useEffect(() => {
    const onGoto = (e: Event) => {
      const i = (e as CustomEvent<number>).detail      // the chips know positions, not ids
      const id = job.scenes.find((s) => s.index === i)?.id
      if (id) setOpen((cur) => new Set(cur).add(id))
    }
    window.addEventListener(GOTO_SCENE, onGoto)
    return () => window.removeEventListener(GOTO_SCENE, onGoto)
  }, [job.scenes])
  // At most one picker is open across the whole panel: a 40-scene job renders one full picker, not 120.
  const [openPicker, setOpenPicker] = useState<{ scene: number; kind: PickerKind } | null>(null)
  const isOpen = (scene: number, kind: PickerKind) => openPicker?.scene === scene && openPicker.kind === kind
  const togglePicker = (scene: number, kind: PickerKind) => setOpenPicker(isOpen(scene, kind) ? null : { scene, kind })
  const update = useUpdateScenes(job.id)
  const approve = useApprove(job.id)
  const regen = useRegenerateScene(job.id)
  const upload = useUploadAsset(job.project_id ?? '')
  const removeVersion = useDeleteVersion(job.id)
  const saveAsset = useSaveSceneAsset(job.id, job.project_id)
  const locks = useSceneLocks(job.id)
  const speak = useSpeakScene(job.id)
  const removeNarration = useRemoveNarration(job.id)
  const structure = useSceneStructure(job.id)
  // The registered edit workflows, so the scene selector needs no hard-coded names.
  const editWorkflows = useWorkflows('edit').data?.workflows ?? []
  const profileEdit = String((job.profile as { edit_workflow?: string } | null)?.edit_workflow ?? 'identity_edit')
  // Which row is asking "delete — and the files too?", and where a new scene is being written.
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null)
  const [addAt, setAddAt] = useState<number | null>(null)
  const [draggingId, setDraggingId] = useState<string | null>(null)
  const [dropAt, setDropAt] = useState<number | null>(null)
  const { kf, clip } = profileSizes(job)
  // What each of the two render buttons will do, so the gate is never a silent blocker.
  const missingClips = job.scenes.filter((s) => !s.video_path).length
  const video_ = (job.profile as { video?: { length?: number; workflow?: string; fps?: number } } | null)?.video
  // What a clip runs to, so a line can be shown against it. The profile is the job's own.
  const clipSecs = video_?.length && video_?.fps ? video_.length / video_.fps : null
  const clipCost = clip && video_?.length
    ? estimateClip({ width: clip[0], height: clip[1], length: video_.length, workflow: video_.workflow })
    : null
  const orphaned = job.status === 'running' && !job.active
  const editable = job.status === 'needs_approval' || job.status === 'done' || job.status === 'failed' || job.status === 'cancelled' || orphaned
  if (!editable) return null
  const video = job.kind === 'video'
  const busy = update.isPending || approve.isPending || regen.isPending || upload.isPending || removeVersion.isPending || saveAsset.isPending
  const slots = job.edit_reference_slots ?? 0
  const faces = assets.filter((a) => a.kind === 'master_face')
  const refAssets = assets.filter((a) => a.kind !== 'master_face' && isImageAsset(a))
  const videoAssets = assets.filter(isVideoAsset)
  /** identity edits (asset id, '' = none) only when the project has faces; else leave untouched */
  const identityEdit = (d: Draft): string | null => (faces.length ? (d.identity_asset_id ?? '') : null)
  // A job in a project picks references as assets (and can upload new ones); otherwise as file paths.
  const assetRefs = refAssets.length > 0 || !!job.project_id
  const refsEdit = (d: Draft): string[] | null => (assetRefs ? d.reference_asset_ids : null)

  const regenerate = (index: number, sceneId: string, what: 'keyframe' | 'video') => {
    const d = drafts[sceneId]
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
        edit_workflow: what === 'keyframe' ? d.edit_workflow : null,
        edit_base: what === 'keyframe' ? d.edit_base : null,
        references: what === 'keyframe' && !assetRefs ? d.references.split('\n').map((x) => x.trim()).filter(Boolean) : null,
        reference_asset_ids: what === 'keyframe' ? refsEdit(d) : null,
        identity_asset_id: job.face_swap ? identityEdit(d) : null,   // a changed face applies to the redone swap
        apply_suggestion: false, // suggestions are applied in the editable prompt above
      },
    }, { onSuccess: () => onStarted?.() })
  }

  /** Send the whole order after a move or a drop: the server renumbers and returns the job. */
  const applyOrder = (ids: string[]) => structure.reorder.mutate(ids)

  const moveScene = (index: number, dir: -1 | 1) => {
    const ids = job.scenes.map((s) => s.id)
    const to = index + dir
    if (to < 0 || to >= ids.length) return
    ;[ids[index], ids[to]] = [ids[to], ids[index]]
    applyOrder(ids)
  }

  /** Drop the dragged scene into gap `at` (0 = before the first row, n = after the last). */
  const dropScene = (at: number) => {
    const ids = job.scenes.map((s) => s.id)
    const from = ids.indexOf(draggingId ?? '')
    setDraggingId(null)
    setDropAt(null)
    if (from < 0) return
    const without = ids.filter((id) => id !== draggingId)
    const target = at > from ? at - 1 : at            // removing the row shifts the later gaps
    if (target === from) return
    without.splice(target, 0, ids[from])
    applyOrder(without)
  }

  const set = (id: string, patch: Partial<Draft>) => setDrafts((d) => ({ ...d, [id]: { ...d[id], ...patch } }))

  /** Unsaved edits in this scene: what the merge effect calls "dirty", per scene rather than
   *  per field — a collapsed row must never hide work you have not saved. */
  const isDirty = (id: string) => {
    const mine = drafts[id], was = saved[id]
    return !!mine && !!was && JSON.stringify(mine) !== JSON.stringify(was)
  }

  /** Picking a version only moves a pointer, so it applies at once — no Save, no regeneration. */
  const selectVersion = (index: number, kind: 'keyframe' | 'clip', id: string) =>
    update.mutate([{ index, ...(kind === 'keyframe' ? { chosen_variant_id: id } : { chosen_clip_id: id }) }])

  const deleteVersion = (index: number, kind: 'keyframe' | 'clip', id: string, label: string) => {
    if (!confirm(`Remove this ${kind} (${label}) from scene ${index + 1}? The file stays on disk; the job stops using it.`)) return
    removeVersion.mutate({ index, versionId: id })
  }

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

  const chooseVideo = (index: number, assetId: string) => {
    const a = assets.find((x) => x.id === assetId)
    const label = a ? (a.label || a.name) : 'the clip'
    if (!confirm(`Use “${label}” as the clip of scene ${index + 1}?\n\nIt is conformed to the job's `
      + `size and fps, and this scene's keyframe becomes its first frame. Any rendered clip stays in the history.`)) return
    setOpenPicker(null)
    regen.mutate({ index, body: { what: 'video', clip_asset_id: assetId, apply_suggestion: false } }, { onSuccess: () => onStarted?.() })
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

  const applySuggestion = (sceneId: string, text: string) => {
    const d = drafts[sceneId]
    if (d.keyframe_prompt.includes(text)) return
    set(sceneId, { keyframe_prompt: `${d.keyframe_prompt.replace(/[.\s]+$/, '')}. Correction: ${text}` })
  }

  const sceneEdits = (): SceneEdit[] =>
    job.scenes.map((s) => {
      const d = drafts[s.id]
      return {
        index: s.index,
        description: d.description,
        keyframe_prompt: d.keyframe_prompt,
        keyframe_negative: d.keyframe_negative,
        animation_prompt: video ? d.animation_prompt : null,
        narration: video ? d.narration : null,
        animation_negative: video ? d.animation_negative : null,
        mode: s.keyframe_path ? null : d.mode,
        edit_workflow: d.edit_workflow,
        edit_base: d.edit_base,
        references: s.keyframe_path || assetRefs ? null : d.references.split('\n').map((x) => x.trim()).filter(Boolean),
        reference_asset_ids: s.keyframe_path ? null : refsEdit(d),
        identity_asset_id: job.face_swap && !s.video_path ? identityEdit(d) : null,
      }
    })
  const save = () => update.mutate(sceneEdits())

  /** Speak a scene's line, saving it first if it has been typed but not saved yet.
   *
   *  The button is enabled on what is in the box, so it has to act on what is in the box —
   *  requiring a separate Save first was a trap: the server reads the saved scene and answered
   *  "no line to speak" for a line plainly on screen. */
  const speakScene = async (index: number, sceneId: string) => {
    if ((drafts[sceneId]?.narration ?? '') !== (job.scenes.find((s) => s.index === index)?.narration ?? '')) {
      try {
        await update.mutateAsync(sceneEdits())
      } catch {
        return      // the save failed and says so in the error box; speaking the OLD line would lie
      }
    }
    speak.mutate({ index })
  }

  return (
    <div className="space-y-3 rounded-lg border border-amber-200 bg-amber-50 p-4">
      {/* Only the title and the actions stay sticky: with the explanation inside it the bar was
          tall enough to cover the top of whichever scene you were reading. */}
      <div className="sticky top-0 z-10 -mx-4 -mt-4 flex flex-wrap items-center justify-between gap-2 border-b border-amber-200 bg-amber-50 px-4 py-2">
        <h2 className="font-semibold">{job.pending_gate ? `Gate: ${job.pending_gate}` : 'Post-run edits'}</h2>
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs text-neutral-600" data-testid="scene-count">
            {job.scenes.length} scene{job.scenes.length === 1 ? '' : 's'} · {open.size} open
          </span>
          <button type="button" data-testid="toggle-all-scenes"
            className="rounded border border-neutral-300 bg-white px-2 py-1 text-xs text-neutral-700 hover:bg-neutral-100"
            onClick={() => setOpen(open.size ? new Set() : new Set(job.scenes.map((s) => s.id)))}>
            {open.size ? 'Collapse all' : 'Expand all'}
          </button>
          <Button onClick={save} disabled={update.isPending}>{update.isPending ? 'Saving…' : 'Save changes'}</Button>
          {job.pending_gate && (
            <Button variant="primary" disabled={approve.isPending} onClick={() => approve.mutate(job.pending_gate!, { onSuccess: () => onStarted?.() })}>
              {approve.isPending ? 'Approving…' : `Approve ${job.pending_gate}`}
            </Button>
          )}
        </div>
      </div>
      <p className="text-sm text-neutral-700">
        {job.pending_gate === 'storyboard' && 'Review the storyboard and prompts, then approve to generate keyframes.'}
        {job.pending_gate === 'keyframes' && (
          <>
            Review the keyframes: regenerate any scene (optionally with an edited prompt), pick a variant,
            tweak animation prompts. <strong>Approve keyframes</strong> renders{' '}
            {missingClips === 0 ? 'nothing — every scene already has a clip' : `the ${missingClips} scene${missingClips === 1 ? '' : 's'} still missing a clip`}
            {' '}(the expensive Wan phase); a scene&apos;s own <strong>Animate this scene</strong> renders just that one, without approving.
          </>
        )}
        {!job.pending_gate && 'Regenerate a keyframe or re-animate a scene (the export folder is rebuilt when the job completes again), or pick a different variant and re-export.'}
      </p>
      <ErrorBox message={update.error ? errorMessage(update.error) : approve.error ? errorMessage(approve.error) : regen.error ? errorMessage(regen.error) : upload.error ? errorMessage(upload.error) : removeVersion.error ? errorMessage(removeVersion.error) : null} />
      {update.isSuccess && !update.isPending && <p className="text-xs text-green-700">Saved.</p>}

      <ErrorBox message={structure.addScene.error ? errorMessage(structure.addScene.error)
        : structure.deleteScene.error ? errorMessage(structure.deleteScene.error)
        : structure.reorder.error ? errorMessage(structure.reorder.error)
        : structure.writePrompts.error ? errorMessage(structure.writePrompts.error) : null} />

      <div className="space-y-4">
        {job.scenes.map((s) => {
          const d = drafts[s.id]
          const locked = !!s.keyframe_path
          const isExp = expanded(s.id)
          const thumb = s.swapped_url ?? s.keyframe_url
          return (
            <div key={s.id}>
            <Gap at={s.index} active={dropAt === s.index && !!draggingId} onOver={setDropAt} onDrop={dropScene}
              onAdd={() => setAddAt(s.index)} adding={addAt === s.index} />
            {addAt === s.index && (
              <AddScene position={s.index} busy={structure.addScene.isPending} onCancel={() => setAddAt(null)}
                onAdd={(b) => structure.addScene.mutate({ position: s.index, ...b }, { onSuccess: () => setAddAt(null) })} />
            )}
            <div id={`scene-${s.index}`} draggable={!busy && !isExp}
              onDragStart={() => setDraggingId(s.id)} onDragEnd={() => { setDraggingId(null); setDropAt(null) }}
              className={`scroll-mt-20 overflow-hidden rounded-lg border-2 bg-white text-sm shadow-sm ${draggingId === s.id ? 'border-blue-400 opacity-60' : 'border-neutral-300'}`}>
              {/* The row header is the whole navigation: a filled number badge scans down the page.
                  It is deliberately NOT sticky when open — floating it over the section hid the
                  description and the negative field behind the number and thumbnail. */}
              <div className={`flex w-full items-center gap-2 px-2 py-2 ${isExp ? 'border-b border-neutral-200 bg-neutral-50' : ''}`}>
                <span className="cursor-grab select-none px-1 text-neutral-400" title="drag to move this scene" aria-hidden>∷</span>
                <button type="button" aria-expanded={isExp} data-testid={`scene-toggle-${s.index}`}
                  onClick={() => toggleScene(s.id)}
                  className="flex min-w-0 flex-1 items-center gap-3 rounded text-left hover:bg-neutral-50">
                  <span className="text-xs text-neutral-400">{isExp ? '▾' : '▸'}</span>
                  <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded bg-neutral-900 text-xs font-semibold text-white">{s.index + 1}</span>
                  {thumb ? (
                    // eslint-disable-next-line @next/next/no-img-element -- local media served by FastAPI
                    <img src={thumb} alt="" loading="lazy" className="h-10 w-10 shrink-0 rounded object-cover" />
                  ) : (
                    <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded border border-dashed border-neutral-300 text-[10px] text-neutral-400">—</span>
                  )}
                  <span className="min-w-0 flex-1 truncate text-neutral-800">{d?.description || s.description || <span className="text-neutral-400">no description</span>}</span>
                  {s.writing_prompts && (
                    <span className="shrink-0 rounded-full bg-blue-100 px-1.5 py-0.5 text-[10px] text-blue-900" data-testid={`writing-prompts-${s.index}`}>writing prompts…</span>
                  )}
                  {isDirty(s.id) && (
                    <span className="shrink-0 rounded-full bg-amber-500 px-1.5 py-0.5 text-[10px] font-medium text-white"
                      data-testid={`scene-dirty-${s.index}`} title="unsaved edits in this scene">unsaved</span>
                  )}
                  <span className="shrink-0 text-xs text-neutral-500">{isExp ? 'collapse' : locked ? 'keyframe made' : 'edit'}</span>
                </button>
                <span className="flex shrink-0 items-center gap-1">
                  <button type="button" aria-label={`move scene ${s.index + 1} up`} disabled={busy || s.index === 0}
                    data-testid={`scene-up-${s.index}`} onClick={() => moveScene(s.index, -1)}
                    className="rounded border border-neutral-300 px-1 text-xs text-neutral-600 disabled:opacity-30">▲</button>
                  <button type="button" aria-label={`move scene ${s.index + 1} down`} disabled={busy || s.index === job.scenes.length - 1}
                    data-testid={`scene-down-${s.index}`} onClick={() => moveScene(s.index, 1)}
                    className="rounded border border-neutral-300 px-1 text-xs text-neutral-600 disabled:opacity-30">▼</button>
                  <button type="button" aria-label={`delete scene ${s.index + 1}`} disabled={busy}
                    data-testid={`scene-delete-${s.index}`} onClick={() => setConfirmDelete(confirmDelete === s.id ? null : s.id)}
                    className="rounded border border-neutral-300 px-1 text-xs text-neutral-600 hover:border-red-300 hover:text-red-700 disabled:opacity-30">✕</button>
                </span>
              </div>
              {confirmDelete === s.id && (
                // The choice is real (a clip costs ten minutes of GPU), so it is spelled out
                // rather than hidden behind a yes/no confirm.
                <div className="flex flex-wrap items-center gap-2 border-b border-red-200 bg-red-50 px-3 py-2 text-xs text-red-900" data-testid={`confirm-delete-${s.index}`}>
                  <span className="font-medium">Delete scene {s.index + 1}?</span>
                  <button type="button" className="rounded border border-red-300 bg-white px-2 py-0.5 font-medium" data-testid={`confirm-delete-keep-${s.index}`}
                    onClick={() => structure.deleteScene.mutate({ index: s.index, deleteFiles: false },
                      { onSuccess: () => setConfirmDelete(null) })}>Remove, keep files</button>
                  <button type="button" className="rounded border border-red-300 bg-white px-2 py-0.5 font-medium" data-testid={`confirm-delete-files-${s.index}`}
                    onClick={() => structure.deleteScene.mutate({ index: s.index, deleteFiles: true },
                      { onSuccess: () => setConfirmDelete(null) })}>Remove and delete its files</button>
                  <button type="button" className="hover:underline" onClick={() => setConfirmDelete(null)}>Cancel</button>
                </div>
              )}
              {isExp && (
              <div className="space-y-3 p-3">
                <div className="space-y-2">
                <label className="block"><span className="text-xs text-neutral-500">description</span>
                  <textarea value={d.description} onChange={(e) => set(s.id, { description: e.target.value })} rows={2} className="mt-0.5 block w-full rounded border border-neutral-300 px-2 py-1" /></label>
                <div className="flex items-center justify-between gap-2">
                  <span className="text-xs text-neutral-500">
                    {s.writing_prompts ? 'Gemma is writing this scene’s prompts…' : 'prompts'}
                  </span>
                  {/* For a scene whose description changed, and for one whose first write failed. */}
                  <button type="button" disabled={busy || s.writing_prompts || !d.description.trim()}
                    data-testid={`write-prompts-${s.index}`} onClick={() => structure.writePrompts.mutate(s.index)}
                    title="rewrite the keyframe and animation prompts from this description, with Gemma"
                    className="text-xs text-neutral-700 underline-offset-2 hover:underline disabled:opacity-40">
                    {s.writing_prompts ? 'writing…' : 'Write prompts with Gemma'}
                  </button>
                </div>
                <label className="block"><span className="text-xs text-neutral-500">keyframe prompt</span>
                  <textarea value={d.keyframe_prompt} onChange={(e) => set(s.id, { keyframe_prompt: e.target.value })} rows={3} className="mt-0.5 block w-full rounded border border-neutral-300 px-2 py-1" /></label>
                <label className="block"><span className="text-xs text-neutral-500">keyframe negative</span>
                  <input value={d.keyframe_negative} onChange={(e) => set(s.id, { keyframe_negative: e.target.value })} className="mt-0.5 block w-full rounded border border-neutral-300 px-2 py-1" /></label>
                {suggestionsFor(s.index).length > 0 && (
                  <div className="space-y-1 rounded bg-amber-50 p-2 text-xs text-amber-900">
                    <div className="font-semibold">Evaluator suggestions{s.keyframe_verdict?.decision === 'retry' ? ' (last verdict: retry)' : ''}</div>
                    {suggestionsFor(s.index).map((text, i) => {
                      const applied = d.keyframe_prompt.includes(text)
                      return (
                        <div key={i} className="flex items-start gap-2">
                          <p className="flex-1">{text}</p>
                          <button type="button" disabled={applied} onClick={() => applySuggestion(s.id, text)}
                            className="shrink-0 rounded border border-amber-300 bg-white px-2 py-0.5 font-medium disabled:opacity-50">
                            {applied ? 'applied' : 'Apply to prompt'}
                          </button>
                        </div>
                      )
                    })}
                    <p className="text-amber-700">Applying appends “Correction: …” to the keyframe prompt, exactly like the automatic retry; edit it, then Regenerate keyframe.</p>
                  </div>
                )}
                {(() => {
                  const hint = s.reference_suggestion
                  const hintAsset = hint && assetRefs ? refAssets.find((a) => a.id === hint.asset_id) : undefined
                  if (!hint || !hintAsset) return null     // a deleted photo leaves no dangling hint
                  const inUse = d.mode === 'edit' && d.reference_asset_ids.includes(hint.asset_id)
                  return (
                    <ReferenceHint scene={s.index} asset={hintAsset} why={hint.why ?? ''} fit={hint.fit ?? 'motif_only'} inUse={inUse}
                      onUse={() => set(s.id, { mode: 'edit', reference_asset_ids: [hint.asset_id, ...d.reference_asset_ids.filter((x) => x !== hint.asset_id)] })}
                      onUndo={() => {
                        const rest = d.reference_asset_ids.filter((x) => x !== hint.asset_id)
                        set(s.id, { reference_asset_ids: rest, mode: rest.length ? 'edit' : 't2i' })
                      }} />
                  )
                })()}
                <div className="space-y-2">
                  {locked && <span className="block text-xs text-neutral-500">mode &amp; references apply on Regenerate keyframe</span>}
                  <div className="flex flex-wrap items-end gap-3">
                  <label className="block w-32 shrink-0"><span className="text-xs text-neutral-500">mode</span>
                    <select value={d.mode} onChange={(e) => set(s.id, { mode: e.target.value as 't2i' | 'edit' })} className="mt-0.5 block w-full rounded border border-neutral-300 bg-white px-2 py-1"><option value="t2i">t2i</option><option value="edit">edit</option></select></label>
                  {d.mode === 'edit' && (() => {
                    // Which edit workflow this scene uses. identity_edit conditions a fresh render
                    // on the references; kv_edit keeps the base image and changes what you describe.
                    const chosen = editWorkflows.find((w) => w.name === (d.edit_workflow || profileEdit))
                    const keepsImage = !!chosen && !chosen.params?.width      // sizes itself from the reference
                    const takesNegative = !chosen || !!chosen.params?.negative
                    return (
                      <div className="min-w-0 flex-1 space-y-1">
                        <label className="flex flex-wrap items-center gap-2">
                          <span className="text-xs text-neutral-500">edit with</span>
                          <select value={d.edit_workflow} data-testid={`edit-workflow-${s.index}`}
                            onChange={(e) => {
                              const name = e.target.value
                              // Editing this scene's own frame is the point of an image-edit
                              // workflow, so tick it by default when one is chosen.
                              const wf = editWorkflows.find((w) => w.name === (name || profileEdit))
                              const image = !!wf && !wf.params?.width
                              set(s.id, { edit_workflow: name, edit_base: image && !!s.keyframe_path ? 'keyframe' : '' })
                            }}
                            className="rounded border border-neutral-300 bg-white px-2 py-1 text-sm">
                            <option value="">profile default ({profileEdit})</option>
                            {editWorkflows.map((w) => (
                              <option key={w.name} value={w.name}>{w.name}{w.description ? ` — ${w.description}` : ''}</option>
                            ))}
                          </select>
                        </label>
                        {keepsImage && s.keyframe_path && (
                          <label className="flex items-center gap-2 text-xs text-neutral-700">
                            <input type="checkbox" checked={d.edit_base === 'keyframe'} data-testid={`edit-base-${s.index}`}
                              onChange={(e) => set(s.id, { edit_base: e.target.checked ? 'keyframe' : '' })} />
                            start from this scene&apos;s current keyframe (describe only the change)
                          </label>
                        )}
                        {!takesNegative && (
                          <p className="text-xs text-amber-700" data-testid={`edit-note-${s.index}`}>
                            {chosen?.name} takes its size from the reference and ignores the keyframe negative.
                          </p>
                        )}
                      </div>
                    )
                  })()}
                  </div>
                  {d.mode === 'edit' && !assetRefs && (
                    <label className="block"><span className="text-xs text-neutral-500">reference paths (one per line)</span>
                      <textarea value={d.references} onChange={(e) => set(s.id, { references: e.target.value })} rows={2} className="mt-0.5 block w-full rounded border border-neutral-300 px-2 py-1 font-mono text-xs" /></label>
                  )}
                  {d.mode === 'edit' && assetRefs && (
                    <div className="min-w-0"><span className="text-xs text-neutral-500">references (project assets, in order){slots > 0 ? ` — the first ${slots} are used` : ''}</span>
                      <div className="flex items-center gap-2">
                        <PickSummary assets={assets} ids={d.reference_asset_ids} empty="no references" />
                        <button type="button" className="shrink-0 text-xs text-neutral-700 hover:underline" data-testid={`refs-${s.index}-choose`}
                          onClick={() => togglePicker(s.index, 'refs')}>{isOpen(s.index, 'refs') ? 'Done' : 'Choose…'}</button>
                      </div>
                      {isOpen(s.index, 'refs') && (
                        <AssetPicker assets={refAssets} selected={d.reference_asset_ids} multi onChange={(ids) => set(s.id, { reference_asset_ids: ids })} testId={`refs-${s.index}`}
                          onUpload={job.project_id ? (f) => uploadThen(f, (id) => setDrafts((all) => ({
                            ...all, [s.id]: { ...all[s.id], reference_asset_ids: [...all[s.id].reference_asset_ids, id] },
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
                        onChange={(ids) => set(s.id, { identity_asset_id: ids[0] ?? null })} />
                    )}
                  </div>
                )}
                {video && (
                  <>
                    <label className="block">
                      <span className="text-xs text-neutral-500">
                        narration (Bulgarian) — plain Cyrillic: the voice handles numbers and stress itself
                      </span>
                      <textarea value={d.narration} onChange={(e) => set(s.id, { narration: e.target.value })} rows={2}
                        data-testid={`narration-${s.index}`}
                        placeholder="Открий своя аромат."
                        className="mt-0.5 block w-full rounded border border-neutral-300 px-2 py-1" />
                    </label>
                    <div className="flex flex-wrap items-center gap-2 text-xs text-neutral-600">
                      <button type="button" disabled={busy || speak.isPending || !d.narration.trim()}
                        data-testid={`speak-${s.index}`}
                        title="speaking saves the line first; it loads a ~5 GB model and frees ComfyUI to make room"
                        onClick={() => { void speakScene(s.index, s.id) }}
                        className="rounded border border-neutral-300 bg-white px-2 py-0.5 disabled:opacity-50">
                        {speak.isPending && speak.variables?.index === s.index ? 'speaking…' : '🔊 Speak'}
                      </button>
                      {s.narration_url && (
                        <>
                          <audio src={s.narration_url} controls preload="none" className="h-8" data-testid={`narration-audio-${s.index}`} />
                          <span data-testid={`narration-fit-${s.index}`}>
                            {s.narration_seconds.toFixed(2)}s
                            {clipSecs != null && ` over a ${clipSecs.toFixed(2)}s clip`}
                            {clipSecs != null && s.narration_seconds > clipSecs && ' — runs into the next scene'}
                          </span>
                          <button type="button" disabled={busy || removeNarration.isPending}
                            data-testid={`narration-remove-${s.index}`}
                            title="throw this take away, so the clip is heard as it was recorded; the line stays, so you can speak it again"
                            onClick={() => removeNarration.mutate({ index: s.index })}
                            className="rounded border border-neutral-300 bg-white px-2 py-0.5 disabled:opacity-50">
                            {removeNarration.isPending && removeNarration.variables?.index === s.index ? 'removing…' : 'Remove voice'}
                          </button>
                        </>
                      )}
                      {removeNarration.isError && removeNarration.variables?.index === s.index && (
                        <span className="text-red-700" data-testid={`narration-remove-error-${s.index}`}>{errorMessage(removeNarration.error)}</span>
                      )}
                      {s.narration_stale && (
                        <span className="rounded bg-amber-100 px-1.5 py-0.5 text-amber-900" data-testid={`narration-stale-${s.index}`}>
                          reworded since it was spoken
                        </span>
                      )}
                      {speak.isError && speak.variables?.index === s.index && (
                        <span className="text-red-700" data-testid={`narration-error-${s.index}`}>{errorMessage(speak.error)}</span>
                      )}
                    </div>
                    <label className="block"><span className="text-xs text-neutral-500">animation prompt{s.video_path ? ' (edits apply on re-animate)' : ''}</span>
                      <textarea value={d.animation_prompt} onChange={(e) => set(s.id, { animation_prompt: e.target.value })} rows={2} className="mt-0.5 block w-full rounded border border-neutral-300 px-2 py-1" /></label>
                    <label className="block"><span className="text-xs text-neutral-500">animation negative</span>
                      <input value={d.animation_negative} onChange={(e) => set(s.id, { animation_negative: e.target.value })} className="mt-0.5 block w-full rounded border border-neutral-300 px-2 py-1" /></label>
                  </>
                )}
                </div>
              {/* Media under the text, keyframe and clip side by side: neither column runs long
                  enough to leave the other empty while scrolling. */}
              <div className={`grid gap-3 ${video ? 'sm:grid-cols-2' : 'sm:max-w-md'}`}>
                <div className="space-y-2" data-testid={`keyframe-cell-${s.index}`}>
                  <MediaThumb url={s.swapped_url ?? s.keyframe_url} width={kf?.[0]} height={kf?.[1]}
                    alt={`scene ${s.index + 1} keyframe`} caption={`Scene ${s.index + 1} · keyframe`} />
                  <div className="flex flex-wrap items-center gap-1">
                    {locked && <Button disabled={busy} onClick={() => regenerate(s.index, s.id, 'keyframe')} title="new keyframe (new seed, current prompts); the previous one stays in the strip below">Regenerate keyframe</Button>}
                    {job.project_id && (
                      <button type="button" disabled={busy} className="text-xs text-neutral-700 underline-offset-2 hover:underline disabled:opacity-50"
                        data-testid={`use-image-${s.index}`} onClick={() => togglePicker(s.index, 'image')}>
                        {isOpen(s.index, 'image') ? 'Cancel' : 'Use my image…'}
                      </button>
                    )}
                    {job.project_id && locked && (
                      // The picker only offers project assets, so a generated frame has to be
                      // copied into the project before another scene can reference it.
                      <button type="button" disabled={busy} className="text-xs text-neutral-700 underline-offset-2 hover:underline disabled:opacity-50"
                        data-testid={`save-asset-${s.index}`} title="copy this keyframe into the project, to use it as a reference in other scenes"
                        onClick={() => saveAsset.mutate({ index: s.index })}>
                        Save as reference
                      </button>
                    )}
                  </div>
                  {saveAsset.isError && saveAsset.variables?.index === s.index && (
                    <p className="text-xs text-red-700" data-testid={`save-asset-error-${s.index}`}>
                      {isMissingRoute(saveAsset.error)
                        ? 'The server does not have this route — restart vorch serve (it is running older code).'
                        : errorMessage(saveAsset.error)}
                    </p>
                  )}
                  {saveAsset.isSuccess && saveAsset.data?.index === s.index && (
                    <p className="text-xs text-green-700" data-testid={`save-asset-done-${s.index}`}>
                      {saveAsset.data.existed
                        ? 'Already in the project — pick it under references.'
                        : 'Saved to the project — pick it under any scene’s references.'}
                    </p>
                  )}
                  <VersionStrip scene={s} kind="keyframe" jobId={job.id} disabled={busy}
                    onSelect={(id) => selectVersion(s.index, 'keyframe', id)}
                    onDelete={(id, label) => deleteVersion(s.index, 'keyframe', id, label)}
                    onRestorePrompt={(prompt, negative) => set(s.id, { keyframe_prompt: prompt, keyframe_negative: negative })} />
                </div>
                {video && (
                  <div className="space-y-2" data-testid={`clip-cell-${s.index}`}>
                    <MediaThumb url={s.video_url} video width={clip?.[0]} height={clip?.[1]}
                      poster={s.swapped_url ?? s.keyframe_url} caption={`Scene ${s.index + 1} · clip`} />
                    <div className="flex flex-wrap items-center gap-1">
                      {locked && (
                        <Button disabled={busy} onClick={() => regenerate(s.index, s.id, 'video')}
                          title={`renders this scene's clip only${clipCost ? ` (${fmtRange(clipCost)} on ${clipCost.where})` : ''}, whatever the gate is waiting for`}>
                          {s.video_path ? 'Re-animate this scene' : 'Animate this scene'}
                        </Button>
                      )}
                      {job.project_id && videoAssets.length > 0 && (
                        // A clip made elsewhere becomes this scene's video, conformed to the
                        // job's size and fps. Nothing downstream can tell the difference.
                        <button type="button" disabled={busy} data-testid={`use-video-${s.index}`}
                          className="text-xs text-neutral-700 underline-offset-2 hover:underline disabled:opacity-50"
                          title="use a video from the project as this scene's clip, instead of animating one"
                          onClick={() => togglePicker(s.index, 'video')}>
                          {isOpen(s.index, 'video') ? 'Cancel' : 'Use my video…'}
                        </button>
                      )}
                      {s.keyframe_url && (
                        // A label the animator redraws comes back as a smear: mark it once and
                        // those pixels are taken from the keyframe instead. No re-render needed.
                        <button type="button" disabled={busy} data-testid={`lock-regions-${s.index}`}
                          className="text-xs text-neutral-700 underline-offset-2 hover:underline disabled:opacity-50"
                          title="mark the label, monogram or emblem the animator must not redraw; it is tracked through the shot"
                          onClick={() => togglePicker(s.index, 'locks')}>
                          {isOpen(s.index, 'locks')
                            ? 'Cancel'
                            : (s.locked_regions?.length
                              ? `Locked: ${s.locked_regions.length} region${s.locked_regions.length > 1 ? 's' : ''}`
                              : 'Lock the label…')}
                        </button>
                      )}
                    </div>
                    {locks.isError && locks.variables?.index === s.index && (
                      <p className="text-xs text-red-700" data-testid={`lock-error-${s.index}`}>
                        {isMissingRoute(locks.error)
                          ? 'The server does not have this route — restart vorch serve (it is running older code).'
                          : errorMessage(locks.error)}
                      </p>
                    )}
                    <VersionStrip scene={s} kind="clip" jobId={job.id} disabled={busy}
                      onSelect={(id) => selectVersion(s.index, 'clip', id)}
                      onDelete={(id, label) => deleteVersion(s.index, 'clip', id, label)}
                      onRestorePrompt={(prompt, negative) => set(s.id, { animation_prompt: prompt, animation_negative: negative })} />
                  </div>
                )}
              </div>
              {isOpen(s.index, 'video') && (
                <div>
                  <span className="text-xs text-neutral-500">
                    pick a clip from the project — it becomes this scene&apos;s video, conformed to the job&apos;s
                    size and fps (nothing is generated, nothing is evaluated)
                  </span>
                  <AssetPicker assets={videoAssets} selected={[]} disabled={busy} testId={`scene-video-${s.index}`}
                    onChange={(ids) => ids[0] && chooseVideo(s.index, ids[0])} />
                </div>
              )}
              {/* Full width: marking a label wants the frame as large as the row allows. */}
              {isOpen(s.index, 'locks') && s.keyframe_url && (
                <LockRegions url={s.swapped_url ?? s.keyframe_url} regions={s.locked_regions ?? []}
                  busy={busy || locks.isPending} testId={`locks-${s.index}`}
                  onCancel={() => togglePicker(s.index, 'locks')}
                  onSave={(regions) => locks.mutate({ index: s.index, regions },
                    { onSuccess: () => setOpenPicker(null) })} />
              )}
              {/* Full width: the picker is a grid of its own and would be cramped in a half column. */}
              {isOpen(s.index, 'image') && (
                <div>
                  <span className="text-xs text-neutral-500">pick a project image or upload one — it replaces the keyframe (not generated, not evaluated)</span>
                  <AssetPicker assets={assets.filter(isImageAsset)} selected={[]} disabled={busy} testId={`keyframe-image-${s.index}`}
                    onChange={(ids) => ids[0] && chooseImage(s.index, ids[0])}
                    onUpload={(f) => uploadThen(f, (id) => chooseImage(s.index, id))} uploading={upload.isPending} />
                </div>
              )}
              </div>
              )}
            </div>
            </div>
          )
        })}
        {/* The last gap: drop a scene at the end, or add one there. */}
        <Gap at={job.scenes.length} active={dropAt === job.scenes.length && !!draggingId} onOver={setDropAt}
          onDrop={dropScene} onAdd={() => setAddAt(job.scenes.length)} adding={addAt === job.scenes.length} />
        {addAt === job.scenes.length && (
          <AddScene position={job.scenes.length} busy={structure.addScene.isPending} onCancel={() => setAddAt(null)}
            onAdd={(b) => structure.addScene.mutate({ position: job.scenes.length, ...b }, { onSuccess: () => setAddAt(null) })} />
        )}
      </div>
    </div>
  )
}
