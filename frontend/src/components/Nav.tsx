'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { Clapperboard, FolderKanban, Folders, Plus, Settings } from 'lucide-react'
import { useConfig } from '@/hooks/useSetup'

const items = [
  { href: '/jobs', label: 'Jobs', icon: FolderKanban },
  { href: '/projects', label: 'Projects', icon: Folders },
  { href: '/jobs/new', label: 'New job', icon: Plus },
  { href: '/setup', label: 'Setup', icon: Settings },
]

export function Nav() {
  const path = usePathname()
  // The UI and the server come from one checkout, so an edited module that the running process
  // predates means new routes answer 404 — which reads as a missing feature, not a stale server.
  const stale = useConfig().data?.code_changed_since_start === true
  return (
    <header className="border-b border-neutral-200 bg-white">
      {stale && (
        <div className="bg-amber-100 px-4 py-1.5 text-center text-xs text-amber-900" data-testid="stale-server">
          <code>vorch serve</code> is running code from before your last change — restart it, or new
          actions will fail with “not found”.
        </div>
      )}
      <div className="mx-auto flex max-w-7xl items-center gap-6 px-4 py-3">
        <Link href="/jobs" className="flex items-center gap-2 font-semibold">
          <Clapperboard className="h-5 w-5" /> vorch
        </Link>
        <nav className="flex gap-1">
          {items.map(({ href, label, icon: Icon }) => {
            const active = href === '/jobs' ? path === '/jobs' || /^\/jobs\/(?!new)/.test(path) : path.startsWith(href)
            return (
              <Link
                key={href}
                href={href}
                className={`flex items-center gap-1.5 rounded px-3 py-1.5 text-sm ${
                  active ? 'bg-neutral-900 text-white' : 'text-neutral-700 hover:bg-neutral-100'
                }`}
              >
                <Icon className="h-4 w-4" /> {label}
              </Link>
            )
          })}
        </nav>
      </div>
    </header>
  )
}
