'use client'

import type { CSSProperties } from 'react'
import { ZoomButton, Zoomable } from '@/components/Lightbox'
import type { LightboxItem } from '@/components/Lightbox'

/** Image or video preview that keeps the job's aspect (portrait stays portrait).
 *  Both open the original in the lightbox: an image on click, a video from its corner ⤢
 *  (a click on the player is play/pause). */
export function MediaThumb({
  url,
  video = false,
  width,
  height,
  alt = '',
  className = '',
  poster,
  caption,
  group,
  groupIndex = 0,
}: {
  url: string | null | undefined
  video?: boolean
  width?: number | null
  height?: number | null
  alt?: string
  className?: string
  /** Still shown before the clip loads — a scene's keyframe, so a clip is never a black box. */
  poster?: string | null
  /** Shown under the enlarged view; defaults to `alt`. */
  caption?: string
  /** Step through these in the lightbox (this preview is `group[groupIndex]`). */
  group?: LightboxItem[]
  groupIndex?: number
}) {
  const style: CSSProperties = width && height ? { aspectRatio: `${width} / ${height}` } : { minHeight: '8rem' }
  if (!url) {
    return (
      <div style={style} className={`flex items-center justify-center rounded bg-neutral-100 text-xs text-neutral-400 ${className}`}>
        {video ? 'no clip yet' : 'no image yet'}
      </div>
    )
  }
  const items = group ?? [{ url, caption: caption ?? alt, video }]
  const index = group ? groupIndex : 0
  if (video) {
    // key on the URL: React reuses <video> elements and would keep playing the old source.
    // `preload="metadata"` fetches only the header, but that is what paints the first frame —
    // with "none" the player is an empty black box until you press play.
    return (
      <div className="relative w-full">
        <video key={url} src={url} controls preload="metadata" poster={poster ?? undefined}
          style={style} className={`w-full rounded bg-black object-contain ${className}`} />
        <ZoomButton items={items} index={index} always className="right-1 top-1" testId="enlarge-video" />
      </div>
    )
  }
  return (
    <Zoomable items={items} index={index} className="w-full">
      {/* eslint-disable-next-line @next/next/no-img-element -- local media served by FastAPI; no optimizer */}
      <img src={url} alt={alt} loading="lazy" style={style} className={`w-full rounded bg-neutral-100 object-contain ${className}`} />
    </Zoomable>
  )
}
