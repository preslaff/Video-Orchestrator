import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, describe, expect, it, vi } from 'vitest'
import ServicesPage from './page'
import type { ServicesReport } from '@/hooks/useServices'

/** The state this machine was actually in: a narration server left running, holding 11 GB. */
const report: ServicesReport = {
  services_file: '/home/x/vorch/services.yaml',
  run_dir: '/home/x/vorch/run',
  log_dir: '/home/x/vorch/logs',
  orphans: ['tts'],
  problems: ['tts: ORPHAN — the vorch that started it is gone'],
  gpu: {
    used: 13833, free: 2001, total: 16303, unattributed_mb: 383,
    holders: [
      { pid: 251592, used_mb: 11268, service: 'tts', cmdline: 'omnivoice.cli.serve', evictable: true, uptime: '11h40m', label: 'tts' },
      { pid: 8536, used_mb: 1960, service: 'gemma', cmdline: 'llama-server', evictable: false, uptime: '23h00m', label: 'gemma' },
      { pid: 777, used_mb: 222, service: null, cmdline: '/usr/bin/mystery-cuda-thing --x', evictable: false, uptime: '2m', label: 'mystery-cuda-thing' },
    ],
  },
  services: [
    { name: 'gemma', description: 'Gemma 4', manage: 'full', lifetime: 'long', status: 'ok', running: true, pid: 8536, pids: [8536], healthy: true, health: {}, vram_mb: 1960, adopted: true, orphan: false, identity: 'verified', identity_detail: 'command line matches', uptime_sec: 82800, idle_sec: null, detail: 'pid 8536; started outside vorch', hint: '', log: '/l/gemma.log', vram: { class: 'resident', needs_mb: 0, holds_mb: 1838 } },
    { name: 'tts', description: 'OmniVoice', manage: 'full', lifetime: 'warm', status: 'fail', running: true, pid: 251592, pids: [251592], healthy: true, health: {}, vram_mb: 11268, adopted: true, orphan: true, identity: 'verified', identity_detail: 'Viktoria/11s.wav', uptime_sec: 42000, idle_sec: null, detail: 'pid 251592; ORPHAN — the vorch that started it is gone', hint: '', log: '/l/tts.log', vram: { class: 'exclusive', needs_mb: 11500, holds_mb: 11268 } },
    { name: 'ui', description: 'Next.js', manage: 'adopt', lifetime: 'long', status: 'down', running: false, pid: null, pids: [], healthy: false, health: {}, vram_mb: null, adopted: false, orphan: false, identity: 'n/a', identity_detail: '', uptime_sec: null, idle_sec: null, detail: 'not running — cd frontend && npm run dev', hint: 'cd frontend && npm run dev', log: '/l/ui.log', vram: { class: 'none', needs_mb: 0, holds_mb: 0 } },
  ],
}

function stub(extra: Record<string, unknown> = {}) {
  vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
    const raw = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url
    const path = new URL(raw, 'http://x').pathname
    const body = path.endsWith('/logs')
      ? { name: 'tts', path: '/l/tts.log', tail: 200, text: 'Launching OmniVoice-BG server: fixed_voice=Viktoria/11s.wav' }
      : path === '/api/setup/config' ? { services_file: report.services_file }
      : { ...report, ...extra }
    return new Response(JSON.stringify(body), { status: 200, headers: { 'content-type': 'application/json' } })
  }))
}

function mount() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(<QueryClientProvider client={qc}><ServicesPage /></QueryClientProvider>)
}

describe('Services panel', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('shows who holds the card, including something it cannot name', async () => {
    stub()
    mount()
    await waitFor(() => expect(screen.getByTestId('gpu-bar')).toBeInTheDocument())
    expect(screen.getByText('2001 MB free')).toBeInTheDocument()
    // A segment per holder plus one for driver overhead.
    expect(screen.getByTestId('gpu-bar').children).toHaveLength(4)
    // An unidentified process is surfaced, not hidden, and flagged as untouchable.
    expect(screen.getByText(/vorch will not touch it/)).toBeInTheDocument()
  })

  it('makes a leaked phase service the loudest thing on the page', async () => {
    stub({
      services: report.services.map((s) =>
        s.name === 'tts' ? { ...s, lifetime: 'phase' as const, detail: 'pid 251592; LEAKED — a phase-scoped service whose vorch is gone' } : s),
      problems: ['tts: LEAKED — a phase-scoped service whose vorch is gone'],
    })
    mount()
    // Named twice on purpose: once in the problems banner, once on its own row.
    await waitFor(() => expect(screen.getAllByText(/LEAKED/)).toHaveLength(2))
    expect(screen.getByText(/should have been stopped when its phase ended/)).toBeInTheDocument()
    expect(screen.getAllByText('fail')).not.toHaveLength(0)
  })

  it('treats a warm service left unattended as a warning, not a leak', async () => {
    // It is still ours and still useful; nothing is left to reap it, which is worth saying once.
    stub({
      services: report.services.map((s) => (s.name === 'tts' ? { ...s, status: 'warn' as const } : s)),
      problems: [],
    })
    mount()
    await waitFor(() => expect(screen.getByText(/nothing is watching it/)).toBeInTheDocument())
    expect(screen.getByText(/stopping it frees the card/)).toBeInTheDocument()
    expect(screen.queryByText(/should have been stopped/)).not.toBeInTheDocument()
  })

  it('does not present an on-demand service that is down as a fault', async () => {
    stub()
    mount()
    await waitFor(() => expect(screen.getByText('down')).toBeInTheDocument())
    expect(screen.getByText(/is not a fault/)).toBeInTheDocument()
    expect(screen.getByText(/vorch\s+will not stop it/)).toBeInTheDocument()
  })

  it('opens a service log on demand', async () => {
    stub()
    mount()
    await waitFor(() => expect(screen.getAllByText('log')).not.toHaveLength(0))
    await userEvent.click(screen.getAllByText('log')[1])
    // The start header is what answers "which voice was this server reading?".
    await waitFor(() => expect(screen.getByText(/fixed_voice=Viktoria\/11s\.wav/)).toBeInTheDocument())
  })

  it('says so plainly when there is no GPU to arbitrate', async () => {
    stub({ gpu: null })
    mount()
    await waitFor(() => expect(screen.getByText(/VRAM is not being arbitrated/)).toBeInTheDocument())
  })
})
