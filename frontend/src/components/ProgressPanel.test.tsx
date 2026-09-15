import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { ProgressPanel, deriveProgress } from './ProgressPanel'
import type { JobPublic } from '@/api/types'
import { describeStep } from '@/api/events'

const scene = (index: number, extra: Partial<JobPublic['scenes'][number]> = {}) => ({
  index, description: '', character: '', setting: '', mood: '', continuous: false, status: 'pending',
  keyframe_prompt: '', animation_prompt: '', keyframe_negative: '', animation_negative: '', mode: 't2i', references: [],
  keyframe_path: null, keyframe_url: null, keyframe_verdict: null, video_path: null, video_url: null, video_verdict: null,
  swapped_path: null, swapped_url: null, swap_verdict: null, variants: [], raw_video_path: null, post_paths: [], identity: '', identity_asset_id: null, identity_url: null, reference_asset_ids: [], ...extra,
})

const base = (over: Partial<JobPublic>): JobPublic => ({
  id: 'j', idea: 'x', style: '', master_face: '', autonomy: 'semi', n_scenes: 2, seed: 0, max_attempts: 2, evaluate: true,
  face_swap: true, mode: 't2i', reference_images: [], project_id: null, profile_id: null, profile: null, kind: 'video',
  export: null, status: 'running', created_at: '', updated_at: '', scenes: [], pending_gate: null, active: false, stopping: false, ...over,
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
  it('describes steps', () => {
    expect(describeStep({ type: 'step', step: 'keyframe', index: 2, variant: 1, variants: 4, attempt: 2, max_attempts: 2, job_id: 'j' }))
      .toBe('Generating keyframe for scene 3 · variant 2/4 · attempt 2/2 — Gemma evaluates each attempt')
    expect(describeStep({ type: 'step', step: 'post', index: 0, chain: ['interp', 'upscale'], job_id: 'j' })).toBe('Post-processing scene 1: interp → upscale')
  })
})
