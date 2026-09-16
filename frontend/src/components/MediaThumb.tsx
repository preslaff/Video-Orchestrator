'use client'

import type { CSSProperties } from 'react'

/** Image or video preview that keeps the job's aspect (portrait stays portrait). */
export function MediaThumb({
  url,
  video = false,
  width,
  height,
  alt = '',
  className = '',
  poster,
}: {
  url: string | null | undefined
  video?: boolean
  width?: number | null
  height?: number | null
  alt?: string
  className?: string
  /** Still shown before the clip loads — a scene's keyframe, so a clip is never a black box. */
  poster?: string | null
}) {
  const style: CSSProperties = width && height ? { aspectRatio: `${width} / ${height}` } : { minHeight: '8rem' }
  if (!url) {
    return (
      <div style={style} className={`flex items-center justify-center rounded bg-neutral-100 text-xs text-neutral-400 ${className}`}>
        {video ? 'no clip yet' : 'no image yet'}
      </div>
    )
  }
  if (video) {
    // key on the URL: React reuses <video> elements and would keep playing the old source.
    // `preload="metadata"` fetches only the header, but that is what paints the first frame —
    // with "none" the player is an empty black box until you press play.
    return <video key={url} src={url} controls preload="metadata" poster={poster ?? undefined}
      style={style} className={`w-full rounded bg-black object-contain ${className}`} />
  }
  // eslint-disable-next-line @next/next/no-img-element -- local media served by FastAPI; no optimizer
  return <img src={url} alt={alt} loading="lazy" style={style} className={`w-full rounded bg-neutral-100 object-contain ${className}`} />
}
