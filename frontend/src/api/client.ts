import createClient from 'openapi-fetch'
import type { paths } from './schema'

/**
 * Where the browser reaches the backend. Empty = same origin (the Next server
 * rewrites /api and /media to FastAPI). Set NEXT_PUBLIC_API_ORIGIN to talk to
 * FastAPI directly (CORS is enabled for the Next origin) — the fallback if a
 * proxied SSE stream ever buffers.
 */
export const apiOrigin = (): string => process.env.NEXT_PUBLIC_API_ORIGIN ?? ''

export const api = createClient<paths>({ baseUrl: apiOrigin() })

/** Human message from the backend's `{ error: { code, message, details } }` envelope. */
export function errorMessage(err: unknown, fallback = 'Request failed'): string {
  if (!err) return fallback
  if (typeof err === 'string') return err
  const e = err as { error?: { message?: string; code?: string }; message?: string; detail?: unknown }
  if (e.error?.message) return e.error.code ? `${e.error.message} (${e.error.code})` : e.error.message
  if (typeof e.detail === 'string') return e.detail
  if (e.message) return e.message
  return fallback
}

export function errorDetails(err: unknown): unknown {
  return (err as { error?: { details?: unknown } } | undefined)?.error?.details
}
