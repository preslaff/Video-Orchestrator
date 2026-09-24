'use client'

import { useQuery } from '@tanstack/react-query'
import { api } from '@/api/client'

/** One process vorch depends on, as the supervisor sees it. */
export interface ServiceState {
  name: string
  description: string
  manage: 'never' | 'adopt' | 'full'
  lifetime: 'long' | 'warm' | 'phase'
  /** `down` is a status, not a fault: for an on-demand service it is the correct state. */
  status: 'ok' | 'warn' | 'fail' | 'down'
  running: boolean
  pid: number | null
  pids: number[]
  healthy: boolean | null
  health: Record<string, unknown>
  vram_mb: number | null
  /** Running, but not started by this vorch — a supported way to work, not a fault. */
  adopted: boolean
  /** Our pidfile, but the vorch that wrote it is gone: memory held by a crash. */
  orphan: boolean
  identity: 'ours' | 'verified' | 'unverified' | 'mismatch' | 'n/a'
  identity_detail: string
  uptime_sec: number | null
  idle_sec: number | null
  detail: string
  hint: string
  log: string
  vram: { class: string; needs_mb: number; holds_mb: number; free_endpoint?: string }
}

export interface GpuHolder {
  pid: number
  used_mb: number
  service: string | null
  cmdline: string
  evictable: boolean
  uptime: string
  label: string
}

export interface ServicesReport {
  services_file: string
  run_dir: string
  log_dir: string
  services: ServiceState[]
  gpu: { used: number; free: number; total: number; unattributed_mb: number; holders: GpuHolder[] } | null
  orphans: string[]
  problems: string[]
}

export function useServices(enabled = true) {
  return useQuery({
    queryKey: ['services'],
    enabled,
    queryFn: async () => {
      const { data, error } = await api.GET('/api/services')
      if (error) throw error
      return data as unknown as ServicesReport
    },
    // Probing walks /proc and makes local HTTP calls, so this is not free — often enough to watch
    // a bring-up, slow enough to leave the page open on.
    refetchInterval: 10_000,
  })
}

export function useServiceLogs(name: string | null, tail = 200) {
  return useQuery({
    queryKey: ['service-logs', name, tail],
    enabled: !!name,
    queryFn: async () => {
      const { data, error } = await api.GET('/api/services/{name}/logs', {
        params: { path: { name: name! }, query: { tail } },
      })
      if (error) throw error
      return data as unknown as { name: string; path: string; tail: number; text: string }
    },
  })
}
