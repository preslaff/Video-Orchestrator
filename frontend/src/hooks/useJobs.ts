'use client'

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '@/api/client'
import { keys } from '@/api/keys'
import type { JobSpec, JobStatus, SceneEdit } from '@/api/types'

export interface JobsFilter {
  status?: JobStatus
  project_id?: string
  limit?: number
  offset?: number
}

export function useJobs(filter: JobsFilter = {}) {
  return useQuery({
    queryKey: keys.jobs(filter),
    queryFn: async () => {
      const { data, error } = await api.GET('/api/jobs', {
        params: { query: { limit: filter.limit ?? 50, offset: filter.offset ?? 0, status: filter.status, project_id: filter.project_id } },
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
