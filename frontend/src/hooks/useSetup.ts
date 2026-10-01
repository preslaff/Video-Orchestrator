'use client'

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '@/api/client'
import { keys } from '@/api/keys'
import type { OutputProfile } from '@/api/types'

// --- environment / config ----------------------------------------------------------

export interface EnvCheck { name: string; status: 'ok' | 'warn' | 'fail'; detail: string; data: Record<string, unknown> }
export interface EnvReport { ok: boolean; status: 'ok' | 'warn' | 'fail'; checked_at: string; duration_ms: number; profile_id: string | null; deep: boolean; checks: EnvCheck[] }

export function useEnv(profileId?: string, deep = false, enabled = true) {
  return useQuery({
    queryKey: keys.env(profileId, deep),
    enabled,
    queryFn: async () => {
      const { data, error } = await api.GET('/api/setup/env', { params: { query: { profile_id: profileId, deep } } })
      if (error) throw error
      return data as unknown as EnvReport
    },
  })
}

export function useRecheckEnv(profileId?: string, deep = false) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async () => {
      const { data, error } = await api.GET('/api/setup/env', { params: { query: { profile_id: profileId, deep, force: true } } })
      if (error) throw error
      return data as unknown as EnvReport
    },
    onSuccess: (report) => qc.setQueryData(keys.env(profileId, deep), report),
  })
}

export type ServerConfig = Record<string, string | number> & {
  started_at?: string
  /** A vorch module on disk is newer than the running process: its routes may be missing. */
  code_changed_since_start?: boolean
}

export function useConfig() {
  return useQuery({
    queryKey: keys.config,
    queryFn: async () => {
      const { data, error } = await api.GET('/api/setup/config')
      if (error) throw error
      return data as ServerConfig
    },
    // Cheap, and it lets the stale-code banner disappear on its own once the server restarts.
    refetchInterval: 30_000,
    // Nav renders this on every page, and a query is stale on mount by default — so anything
    // that remounts the tree re-requests it. The interval already keeps it fresh enough.
    staleTime: 30_000,
  })
}

// --- workflows -------------------------------------------------------------------------

export interface WorkflowParam { targets: { node: string; field: string }[]; type: string; default: unknown; required: boolean; constraint: string | null; min: number | null; max: number | null; enum_from: { class_type: string; input: string } | null; from_aspect: Record<string, string> | null }
export interface WorkflowInfo { name: string; kind: string; template: string; description: string; tags: string[]; timeout: number | null; backend: string; builtin: boolean; cost: { min_usd: number; max_usd: number } | null; source: string | null; inputs: Record<string, string>; outputs: { type: string | null; nodes: string[] }; params: Record<string, WorkflowParam>; requires: { nodes: string[]; models: { node: string; class_type: string; field: string; value: string }[] }; problems?: string[] }

export function useWorkflows(kind?: string) {
  return useQuery({
    queryKey: [...keys.workflows, kind ?? null],
    queryFn: async () => {
      const { data, error } = await api.GET('/api/workflows', { params: { query: { kind } } })
      if (error) throw error
      return data as unknown as { workflows: WorkflowInfo[]; errors: Record<string, string>; kinds: string[]; user_dir: string }
    },
  })
}

export function useWorkflow(name: string | undefined) {
  return useQuery({
    queryKey: keys.workflow(name ?? ''),
    enabled: !!name,
    queryFn: async () => {
      const { data, error } = await api.GET('/api/workflows/{name}', { params: { path: { name: name! } } })
      if (error) throw error
      return data as unknown as WorkflowInfo
    },
  })
}

// --- profiles -------------------------------------------------------------------------

export function useProfiles() {
  return useQuery({
    queryKey: keys.profiles,
    queryFn: async () => {
      const { data, error } = await api.GET('/api/profiles')
      if (error) throw error
      return data
    },
  })
}

export function useProfile(id: string | undefined) {
  return useQuery({
    queryKey: keys.profile(id ?? ''),
    enabled: !!id,
    queryFn: async () => {
      const { data, error } = await api.GET('/api/profiles/{profile_id}', { params: { path: { profile_id: id! } } })
      if (error) throw error
      return data
    },
  })
}

function invalidateProfiles(qc: ReturnType<typeof useQueryClient>) {
  void qc.invalidateQueries({ queryKey: keys.profiles })
  void qc.invalidateQueries({ queryKey: ['profile'] })
}

export function useCreateProfile() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (body: { name: string; base?: string | null; overrides?: Record<string, unknown> | null }) => {
      const { data, error } = await api.POST('/api/profiles', { body })
      if (error) throw error
      return data as OutputProfile
    },
    onSuccess: () => invalidateProfiles(qc),
  })
}

export function useUpdateProfile(id: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (patch: Record<string, unknown>) => {
      const { data, error } = await api.PUT('/api/profiles/{profile_id}', { params: { path: { profile_id: id } }, body: patch })
      if (error) throw error
      return data as OutputProfile
    },
    onSuccess: () => invalidateProfiles(qc),
  })
}

export function useDeleteProfile() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (id: string) => {
      const { error } = await api.DELETE('/api/profiles/{profile_id}', { params: { path: { profile_id: id } } })
      if (error) throw error
    },
    onSuccess: () => invalidateProfiles(qc),
  })
}

export function useCloneProfile() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async ({ id, name, overrides }: { id: string; name: string; overrides?: Record<string, unknown> | null }) => {
      const { data, error } = await api.POST('/api/profiles/{profile_id}/clone', { params: { path: { profile_id: id } }, body: { name, overrides } })
      if (error) throw error
      return data as OutputProfile
    },
    onSuccess: () => invalidateProfiles(qc),
  })
}

// --- projects ----------------------------------------------------------------------------

export function useProjects() {
  return useQuery({
    queryKey: keys.projects,
    queryFn: async () => {
      const { data, error } = await api.GET('/api/projects')
      if (error) throw error
      return data
    },
  })
}

export function useProject(id: string | undefined) {
  return useQuery({
    queryKey: keys.project(id ?? ''),
    enabled: !!id,
    queryFn: async () => {
      const { data, error } = await api.GET('/api/projects/{project_id}', { params: { path: { project_id: id! } } })
      if (error) throw error
      return data
    },
    // Poll only while Gemma is describing assets in the background, so labels appear on their own.
    refetchInterval: (query) =>
      query.state.data?.assets.some((a) => a.describe_state === 'queued' || a.describe_state === 'describing') ? 3000 : false,
  })
}

function invalidateProjects(qc: ReturnType<typeof useQueryClient>, id?: string) {
  void qc.invalidateQueries({ queryKey: keys.projects })
  if (id) void qc.invalidateQueries({ queryKey: keys.project(id) })
}

export function useCreateProject() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (input: { name: string; slug?: string | null; brand_notes?: string; style_notes?: string; default_profile_id?: string; export_root?: string; describe_hint?: string }) => {
      const body = { brand_notes: '', style_notes: '', default_profile_id: 'cinematic_16x9', export_root: '', describe_hint: '', slug: null, ...input }
      const { data, error } = await api.POST('/api/projects', { body })
      if (error) throw error
      return data
    },
    onSuccess: () => invalidateProjects(qc),
  })
}

export function useUpdateProject(id: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (patch: { name?: string | null; brand_notes?: string | null; style_notes?: string | null; default_profile_id?: string | null; export_root?: string | null; describe_hint?: string | null }) => {
      const { data, error } = await api.PUT('/api/projects/{project_id}', { params: { path: { project_id: id } }, body: patch })
      if (error) throw error
      return data
    },
    onSuccess: () => invalidateProjects(qc, id),
  })
}

export function useDeleteProject() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async ({ id, deleteFiles = false }: { id: string; deleteFiles?: boolean }) => {
      const { error } = await api.DELETE('/api/projects/{project_id}', { params: { path: { project_id: id }, query: { delete_files: deleteFiles } } })
      if (error) throw error
    },
    onSuccess: () => invalidateProjects(qc),
  })
}

export function useUploadAsset(projectId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async ({ file, kind, label }: { file: File; kind: string; label?: string }) => {
      const form = new FormData()
      form.append('file', file)
      form.append('kind', kind)
      form.append('label', label ?? '')
      const res = await fetch(`/api/projects/${projectId}/assets`, { method: 'POST', body: form })
      const body = await res.json()
      if (!res.ok) throw body
      return body
    },
    onSuccess: () => invalidateProjects(qc, projectId),
  })
}

type AssetKind = 'master_face' | 'reference' | 'product' | 'logo' | 'other' | 'video'

/** Change an asset's label (it becomes a person's) and/or its kind (a suggested label is rebuilt). */
export function useUpdateAsset(projectId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async ({ assetId, label, kind }: { assetId: string; label?: string; kind?: AssetKind }) => {
      const { data, error } = await api.PATCH('/api/projects/{project_id}/assets/{asset_id}', {
        params: { path: { project_id: projectId, asset_id: assetId } },
        body: { label: label ?? null, kind: kind ?? null },
      })
      if (error) throw error
      return data
    },
    onSuccess: () => invalidateProjects(qc, projectId),
  })
}

/** Queue assets for background description: unlabelled ones, or given ids; ``force`` also redoes suggestions. */
export function useDescribeAssets(projectId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async ({ assetIds, force = false }: { assetIds?: string[]; force?: boolean } = {}) => {
      const { data, error } = await api.POST('/api/projects/{project_id}/assets/describe', {
        params: { path: { project_id: projectId } },
        body: { asset_ids: assetIds ?? null, force },
      })
      if (error) throw error
      return data
    },
    onSuccess: () => invalidateProjects(qc, projectId),
  })
}

export function useDeleteAsset(projectId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (assetId: string) => {
      const { error } = await api.DELETE('/api/projects/{project_id}/assets/{asset_id}', { params: { path: { project_id: projectId, asset_id: assetId } } })
      if (error) throw error
    },
    onSuccess: () => invalidateProjects(qc, projectId),
  })
}
