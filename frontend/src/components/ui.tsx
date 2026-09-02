'use client'

import type { ReactNode } from 'react'
import type { JobStatus } from '@/api/types'

export function Card({ title, children, actions, className = '' }: { title?: ReactNode; children: ReactNode; actions?: ReactNode; className?: string }) {
  return (
    <section className={`rounded-lg border border-neutral-200 bg-white p-4 shadow-sm ${className}`}>
      {(title || actions) && (
        <div className="mb-3 flex items-center justify-between gap-3">
          {title && <h2 className="text-sm font-semibold uppercase tracking-wide text-neutral-600">{title}</h2>}
          {actions}
        </div>
      )}
      {children}
    </section>
  )
}

const STATUS_STYLES: Record<string, string> = {
  pending: 'bg-neutral-200 text-neutral-800',
  running: 'bg-blue-100 text-blue-800',
  needs_approval: 'bg-amber-100 text-amber-900',
  done: 'bg-green-100 text-green-800',
  failed: 'bg-red-100 text-red-800',
  cancelled: 'bg-neutral-100 text-neutral-500',
}

export function StatusBadge({ status }: { status: JobStatus | string }) {
  return (
    <span className={`inline-flex items-center gap-1 rounded px-2 py-0.5 text-xs font-medium ${STATUS_STYLES[status] ?? 'bg-neutral-100'}`}>
      {status === 'running' && <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-blue-600" />}
      {status.replace('_', ' ')}
    </span>
  )
}

export function Button({
  children,
  onClick,
  variant = 'default',
  disabled,
  type = 'button',
  title,
  ...rest
}: {
  children: ReactNode
  onClick?: () => void
  variant?: 'default' | 'primary' | 'danger' | 'ghost'
  disabled?: boolean
  type?: 'button' | 'submit'
  title?: string
} & Record<`data-${string}`, string>) {
  const styles = {
    default: 'border border-neutral-300 bg-white hover:bg-neutral-50',
    primary: 'bg-neutral-900 text-white hover:bg-neutral-700',
    danger: 'border border-red-300 text-red-700 hover:bg-red-50',
    ghost: 'text-neutral-600 hover:bg-neutral-100',
  }[variant]
  return (
    <button type={type} onClick={onClick} disabled={disabled} title={title} {...rest}
      className={`rounded px-3 py-1.5 text-sm font-medium disabled:cursor-not-allowed disabled:opacity-50 ${styles}`}>
      {children}
    </button>
  )
}

export function ErrorBox({ message }: { message: string | null | undefined }) {
  if (!message) return null
  return <div className="rounded border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800">{message}</div>
}

export function Empty({ children }: { children: ReactNode }) {
  return <p className="py-6 text-center text-sm text-neutral-500">{children}</p>
}

export function fmtTime(iso: string | null | undefined): string {
  if (!iso) return ''
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString()
}
