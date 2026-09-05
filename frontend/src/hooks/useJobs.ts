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
