import type { components } from './schema'

type S = components['schemas']

export type JobPublic = S['JobPublic']
export type ScenePublic = S['ScenePublic']
export type VariantOut = S['VariantOut']
export type VerdictOut = S['VerdictOut']
export type JobSummary = S['JobSummaryOut']
export type JobSpec = S['JobSpec']
export type DecisionOut = S['DecisionOut']
export type ProvenanceOut = S['ProvenanceOut']
export type OutputProfile = S['OutputProfile']
export type ProjectOut = S['ProjectOut']
export type ProjectSummary = S['ProjectSummaryOut']
export type AssetOut = S['AssetOut']
export type SceneEdit = S['SceneEdit']
export type SceneRegenerate = S['SceneRegenerate']
export type Gate = 'storyboard' | 'keyframes'
export type JobStatus = 'pending' | 'running' | 'needs_approval' | 'done' | 'failed' | 'cancelled'
export type AssemblySpecIn = S['AssemblySpecIn']
export type AssemblyStatus = S['AssemblyStatus']
export type AssemblyClip = S['AssemblyClipOut']
export type AssemblyRecord = S['AssemblyRecordOut']
export type AssemblyCut = S['AssemblyCutIn']
