import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { ProgressPanel, deriveProgress } from './ProgressPanel'
import type { JobPublic } from '@/api/types'
import { describeStep } from '@/api/events'

const scene = (index: number, extra: Partial<JobPublic['scenes'][number]> = {}) => ({
  index, id: `s${index}`, writing_prompts: false, description: '', character: '', setting: '', mood: '', continuous: false, status: 'pending',
  keyframe_prompt: '', animation_prompt: '', keyframe_negative: '', animation_negative: '', mode: 't2i', edit_workflow: '', keyframe_workflow: '', edit_base: '', edit_base_path: '', references: [],
  keyframe_path: null, keyframe_url: null, keyframe_verdict: null, video_path: null, video_url: null, video_verdict: null,
  swapped_path: null, swapped_url: null, swap_verdict: null, variants: [], raw_video_path: null, post_paths: [], identity: '', identity_asset_id: null, identity_url: null, reference_asset_ids: [], keyframe_source: 'generated', keyframe_asset_id: null, clips: [], locked_regions: [], narration: '', narration_url: null, narration_seconds: 0, narration_stale: false, texts: [], ...extra,
})

const base = (over: Partial<JobPublic>): JobPublic => ({
  id: 'j', idea: 'x', style: '', master_face: '', autonomy: 'semi', n_scenes: 2, seed: 0, max_attempts: 2, evaluate: true,
  face_swap: true, mode: 't2i', reference_images: [], project_id: null, profile_id: null, profile: null, kind: 'video',
  export: null, status: 'running', created_at: '', updated_at: '', scenes: [], pending_gate: null, active: true, stopping: false, budget_usd: null, spend_usd: 0, spend_calls: 0, spend: [], budget_reason: '', edit_reference_slots: 2, ...over,
})

describe('deriveProgress', () => {
  it('starts in the storyboard phase with nothing done', () => {
    const p = deriveProgress(base({}), [])
    expect(p.current).toBe('storyboard')
    expect(p.done).toEqual([])
    expect(p.percent).toBe(0)
  })
  it('counts keyframes and clips as units and follows phase events', () => {
    const job = base({ scenes: [scene(0, { keyframe_path: '/a', status: 'keyframe' }), scene(1)] })
    const p = deriveProgress(job, [{ type: 'phase', phase: 'keyframes', job_id: 'j' }, { type: 'step', step: 'keyframe', index: 1, job_id: 'j' }])
    expect(p.done).toEqual(['storyboard'])
    expect(p.current).toBe('keyframes')
    expect(p.units).toEqual({ done: 1, total: 4 })
    expect(p.step?.index).toBe(1)
  })
  it('marks everything done for a finished, exported video job', () => {
    const job = base({ status: 'done', export: { ok: true }, scenes: [scene(0, { keyframe_path: '/a', video_path: '/v', status: 'complete' })] })
    const p = deriveProgress(job, [])
    expect(p.done).toEqual(['storyboard', 'keyframes', 'videos', 'export'])
    expect(p.percent).toBe(100)
    expect(p.current).toBeNull()
  })
  it('uses the stills phase list', () => {
    expect(deriveProgress(base({ kind: 'stills' }), []).phases).toEqual(['storyboard', 'keyframes', 'stills', 'export'])
  })
})

describe('ProgressPanel', () => {
  it('shows the current activity and the gate state', () => {
    const job = base({ scenes: [scene(0), scene(1)] })
    render(<ProgressPanel job={job} events={[{ type: 'phase', phase: 'keyframes', job_id: 'j' }, { type: 'step', step: 'animate', index: 0, length: 81, fps: 16, seconds: 5.06, job_id: 'j', _at: Date.now() - 65_000 }]} connected />)
    expect(screen.getByText(/Animating scene 1 with Wan — 81 frames @ 16 fps/)).toBeInTheDocument()
    expect(screen.getByText(/1m 05s/)).toBeInTheDocument()
    expect(screen.getByText('● live')).toBeInTheDocument()
    render(<ProgressPanel job={base({ status: 'needs_approval', scenes: [scene(0, { keyframe_path: '/a', status: 'keyframe' })] })} events={[]} connected={false} />)
    expect(screen.getByText(/Videos · awaiting approval/)).toBeInTheDocument()
  })
  it('shows stopping, stopped and orphaned jobs', () => {
    const { unmount } = render(<ProgressPanel job={base({ stopping: true })} events={[]} connected />)
    expect(screen.getByText('stopping…')).toBeInTheDocument()
    expect(screen.getByText(/Stopping — finishing the current step/)).toBeInTheDocument()
    unmount()
    const stopped = render(<ProgressPanel job={base({ status: 'cancelled', active: false })} events={[]} connected={false} />)
    expect(screen.getByText('stopped')).toBeInTheDocument()
    expect(screen.getByText(/Edit any scene below, then Resume/)).toBeInTheDocument()
    stopped.unmount()
    render(<ProgressPanel job={base({ status: 'running', active: false })} events={[]} connected={false} />)
    expect(screen.getByText('not running')).toBeInTheDocument()
    expect(screen.queryByText('● live')).toBeNull()
  })
  it('describes steps', () => {
    expect(describeStep({ type: 'step', step: 'keyframe', index: 2, variant: 1, variants: 4, attempt: 2, max_attempts: 2, job_id: 'j' }))
      .toBe('Generating keyframe for scene 3 · variant 2/4 · attempt 2/2 — Gemma evaluates each attempt')
    expect(describeStep({ type: 'step', step: 'post', index: 0, chain: ['interp', 'upscale'], job_id: 'j' })).toBe('Post-processing scene 1: interp → upscale')
  })
})

describe('ProgressPanel with a long-lived page', () => {
  it('scans a large timeline without copying it', () => {
    // The panel re-renders every second while a job runs; the old code copied and reversed
    // the whole array three times per render, which is what made a page left open all evening
    // spin the browser's GC.
    const many = Array.from({ length: 20_000 }, (_, i) => ({ type: 'step', step: `s${i}`, index: 0, job_id: 'j' }))
    // Frozen: a copy-and-reverse would mutate nothing, but scanning must not need one.
    const frozen = Object.freeze([...many, { type: 'phase', phase: 'videos', job_id: 'j' }])
    const j = base({ scenes: [scene(0), scene(1)] })
    const t0 = performance.now()
    const p = deriveProgress(j, frozen)
    expect(performance.now() - t0).toBeLessThan(50)
    expect(p.current).toBe('videos')
    expect(p.step).not.toBeNull()
  })
})
