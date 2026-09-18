'use client'

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '@/api/client'
import { keys } from '@/api/keys'
import type { AssemblySpecIn, AssemblyStatus, JobSpec, JobStatus, SceneEdit, SceneRegenerate } from '@/api/types'

export interface JobsFilter {
  status?: JobStatus
  project_id?: string
  unassigned?: boolean
  limit?: number
  offset?: number
}

export function useJobs(filter: JobsFilter = {}) {
  return useQuery({
    queryKey: keys.jobs(filter),
    queryFn: async () => {
      const { data, error } = await api.GET('/api/jobs', {
        params: { query: { limit: filter.limit ?? 50, offset: filter.offset ?? 0, status: filter.status, project_id: filter.project_id, unassigned: filter.unassigned || undefined } },
      })
      if (error) throw error
      return data
    },
  })
}

export function useJob(id: string | undefined) {
  return useQuery({
    queryKey: keys.job(id ?? ''),
    enabled: !!id,
    queryFn: async () => {
      const { data, error } = await api.GET('/api/jobs/{job_id}', { params: { path: { job_id: id! } } })
      if (error) throw error
      return data
    },
    // Live updates come over SSE; this is the safety net for when that stream is not there
    // (a dropped connection, a proxy timeout), so a finished segment never waits for F5.
    refetchInterval: (query) =>
      query.state.data?.status === 'running' || query.state.data?.status === 'pending' ? 5000 : false,
  })
}

export function useDecisions(id: string | undefined) {
  return useQuery({
    queryKey: keys.decisions(id ?? ''),
    enabled: !!id,
    queryFn: async () => {
      const { data, error } = await api.GET('/api/jobs/{job_id}/decisions', { params: { path: { job_id: id! } } })
      if (error) throw error
      return data
    },
  })
}

export function useProvenance(id: string | undefined) {
  return useQuery({
    queryKey: keys.provenance(id ?? ''),
    enabled: !!id,
    queryFn: async () => {
      const { data, error } = await api.GET('/api/jobs/{job_id}/provenance', { params: { path: { job_id: id! } } })
      if (error) throw error
      return data
    },
  })
}

export function useExportStatus(id: string | undefined) {
  return useQuery({
    queryKey: keys.exportStatus(id ?? ''),
    enabled: !!id,
    queryFn: async () => {
      const { data, error } = await api.GET('/api/jobs/{job_id}/export', { params: { path: { job_id: id! } } })
      if (error) throw error
      return data as Record<string, unknown>
    },
  })
}

export function useCreateJob() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (spec: JobSpec) => {
      const { data, error } = await api.POST('/api/jobs', { body: spec })
      if (error) throw error
      return data
    },
    onSuccess: () => void qc.invalidateQueries({ queryKey: ['jobs'] }),
  })
}

export function useApprove(id: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (gate: 'storyboard' | 'keyframes') => {
      const { data, error } = await api.POST('/api/jobs/{job_id}/approve/{gate}', { params: { path: { job_id: id, gate } } })
      if (error) throw error
      return data
    },
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: keys.job(id) })
      void qc.invalidateQueries({ queryKey: ['jobs'] })
    },
  })
}

export function useRetry(id: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async () => {
      const { data, error } = await api.POST('/api/jobs/{job_id}/retry', { params: { path: { job_id: id } } })
      if (error) throw error
      return data
    },
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: keys.job(id) })
      void qc.invalidateQueries({ queryKey: ['jobs'] })
    },
  })
}

/** Ask a running job to stop at its next safe point; it ends `cancelled` ("stopped"). */
export function useStopJob(id: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async () => {
      const { data, error } = await api.POST('/api/jobs/{job_id}/stop', { params: { path: { job_id: id } } })
      if (error) throw error
      return data
    },
    onSuccess: () => void qc.invalidateQueries({ queryKey: keys.job(id) }),
  })
}

export function useUpdateScenes(id: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (scenes: SceneEdit[]) => {
      const { data, error } = await api.PUT('/api/jobs/{job_id}/scenes', { params: { path: { job_id: id } }, body: { scenes } })
      if (error) throw error
      return data
    },
    onSuccess: () => void qc.invalidateQueries({ queryKey: keys.job(id) }),
  })
}

export function useDeleteJob() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (id: string) => {
      const { error } = await api.DELETE('/api/jobs/{job_id}', { params: { path: { job_id: id } } })
      if (error) throw error
    },
    onSuccess: () => void qc.invalidateQueries({ queryKey: ['jobs'] }),
  })
}

export function useRunExport(id: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (force: boolean) => {
      const { data, error } = await api.POST('/api/jobs/{job_id}/export', { params: { path: { job_id: id }, query: { force } } })
      if (error) throw error
      return data as Record<string, unknown>
    },
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: keys.exportStatus(id) })
      void qc.invalidateQueries({ queryKey: keys.job(id) })
    },
  })
}

export function useRegenerateScene(id: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async ({ index, body }: { index: number; body: SceneRegenerate }) => {
      const { data, error } = await api.POST('/api/jobs/{job_id}/scenes/{index}/regenerate', {
        params: { path: { job_id: id, index } },
        body,
      })
      if (error) throw error
      return data
    },
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: keys.job(id) })
      void qc.invalidateQueries({ queryKey: ['jobs'] })
    },
  })
}

/** Drop one keyframe or clip version from a scene's history (the file stays on disk). */
export function useDeleteVersion(id: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async ({ index, versionId }: { index: number; versionId: string }) => {
      const { data, error } = await api.DELETE('/api/jobs/{job_id}/scenes/{index}/versions/{version_id}', {
        params: { path: { job_id: id, index, version_id: versionId } },
      })
      if (error) throw error
      return data
    },
    onSuccess: () => void qc.invalidateQueries({ queryKey: keys.job(id) }),
  })
}

/** Copy a scene's keyframe into the job's project as an asset, to reuse it as a reference
 *  in other scenes (the reference picker only offers project assets). */
export function useSaveSceneAsset(id: string, projectId?: string | null) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async ({ index, versionId, label }: { index: number; versionId?: string; label?: string }) => {
      const { data, error, response } = await api.POST('/api/jobs/{job_id}/scenes/{index}/save-asset', {
        params: { path: { job_id: id, index } },
        body: { version_id: versionId ?? null, kind: 'reference', label: label ?? '' },
      })
      if (error) throw error
      // 200 = the project already held these bytes; 201 = a new asset was copied in.
      return { asset: data, existed: response.status === 200, index }
    },
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: keys.projects })
      if (projectId) void qc.invalidateQueries({ queryKey: keys.project(projectId) })
    },
  })
}

export function useAssembly(id: string | undefined) {
  return useQuery({
    queryKey: keys.assembly(id ?? ''),
    enabled: !!id,
    queryFn: async () => {
      const { data, error } = await api.GET('/api/jobs/{job_id}/assemble', { params: { path: { job_id: id! } } })
      if (error) throw error
      return data as AssemblyStatus
    },
  })
}

export function useRunAssembly(id: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async ({ spec, force = false }: { spec: AssemblySpecIn; force?: boolean }) => {
      const { data, error } = await api.POST('/api/jobs/{job_id}/assemble', {
        params: { path: { job_id: id }, query: { force } },
        body: spec,
      })
      if (error) throw error
      return data as AssemblyStatus
    },
    onSuccess: (data) => {
      qc.setQueryData(keys.assembly(id), data)
      void qc.invalidateQueries({ queryKey: keys.job(id) })
    },
  })
}

export function useSelectAssemblyVersion(id: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (version: number) => {
      const { data, error } = await api.POST('/api/jobs/{job_id}/assemble/versions/{version}/select', {
        params: { path: { job_id: id, version } },
      })
      if (error) throw error
      return data as AssemblyStatus
    },
    onSuccess: (data) => {
      qc.setQueryData(keys.assembly(id), data)
      void qc.invalidateQueries({ queryKey: keys.job(id) })
      void qc.invalidateQueries({ queryKey: keys.exportStatus(id) })
    },
  })
}

export function useDeleteAssemblyVersion(id: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (version: number) => {
      const { data, error } = await api.DELETE('/api/jobs/{job_id}/assemble/versions/{version}', {
        params: { path: { job_id: id, version } },
      })
      if (error) throw error
      return data as AssemblyStatus
    },
    onSuccess: (data) => {
      qc.setQueryData(keys.assembly(id), data)
      void qc.invalidateQueries({ queryKey: keys.job(id) })
      void qc.invalidateQueries({ queryKey: keys.exportStatus(id) })
    },
  })
}
