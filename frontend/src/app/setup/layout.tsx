'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'

const tabs = [
  { href: '/setup/env', label: 'Environment' },
  { href: '/setup/workflows', label: 'Workflows' },
  { href: '/setup/profiles', label: 'Profiles' },
  { href: '/setup/projects', label: 'Projects' },
]

export default function SetupLayout({ children }: { children: React.ReactNode }) {
  const path = usePathname()
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-4">
        <h1 className="text-xl font-semibold">Setup</h1>
        <nav className="flex gap-1 text-sm">
          {tabs.map((t) => (
            <Link key={t.href} href={t.href} className={`rounded px-3 py-1 ${path.startsWith(t.href) ? 'bg-neutral-200 font-medium' : 'hover:bg-neutral-100'}`}>{t.label}</Link>
          ))}
        </nav>
      </div>
      {children}
    </div>
  )
}
