'use client'

import { useEffect, useRef, useState } from 'react'
import type { TextItem } from '@/api/types'
import { ANCHORS, FALLBACK_EFFECTS, anchorStyle, blankText, textStyle, visibleAt } from '@/lib/titles'
import { Button } from '@/components/ui'

/**
 * Captions over a piece of video — the editor and a live preview.
 *
 * Controlled: the whole list goes back on every change, so the parent decides when it is saved
 * (the timeline panel submits it with the spec; a scene saves it through its own route).
 *
 * The preview is honest about what it is. The **timing** is exact, because it is the same
 * arithmetic the renderer uses. The **metrics** are not: libass picks fonts through fontconfig,
 * shapes with harfbuzz and strokes a real outline, none of which a browser reproduces, so line
 * breaks and the exact weight of an outline will differ slightly from the burn-in.
 */
/**
 * A number field that lets you type a decimal.
 *
 * A controlled `<input type="number">` that reformats its value on every keystroke cannot accept
 * "7.5": the moment "7." is parsed it becomes 7, the dot is gone, and the next key makes 75. So
 * the raw text is held here while the field has focus and the parsed value is reported upwards —
 * the same reason `AssemblePanel` keeps its trim and target fields as raw strings.
 */
function NumField({
  value, onCommit, label, testId, min, max, step = 0.1, width = 'w-16', disabled, suffix,
}: {
  value: number
  onCommit: (v: number) => void
  label: string
  testId?: string
  min?: number
  max?: number
  step?: number
  width?: string
  disabled?: boolean
  suffix?: React.ReactNode
}) {
  const [raw, setRaw] = useState<string | null>(null)
  const shown = raw ?? String(Math.round(value * 1e4) / 1e4)
  const clamp = (v: number) => Math.min(max ?? Infinity, Math.max(min ?? -Infinity, v))
  return (
    <>
      <input type="number" min={min} max={max} step={step} value={shown} disabled={disabled}
        aria-label={label} data-testid={testId}
        onChange={(e) => {
          setRaw(e.target.value)
          const n = parseFloat(e.target.value)
          if (Number.isFinite(n)) onCommit(clamp(n))
        }}
        onBlur={() => setRaw(null)}
        className={`${width} rounded border border-neutral-300 px-1 py-0.5`} />
      {suffix}
    </>
  )
}


export function TextItems({
  items, onChange, videoUrl, effects, presets, limit, busy = false, testId = 'texts',
}: {
  items: TextItem[]
  onChange: (items: TextItem[]) => void
  videoUrl?: string | null
  effects?: string[]
  presets?: Record<string, Record<string, unknown>>
  /** How long the thing underneath lasts, so a caption cannot be scheduled past its end. */
  limit?: number | null
  busy?: boolean
  testId?: string
}) {
  const fx = effects?.length ? effects : FALLBACK_EFFECTS
  const [at, setAt] = useState(0)
  const [boxH, setBoxH] = useState(0)
  const box = useRef<HTMLDivElement>(null)

  // The preview scales with the element exactly as libass scales with PlayRes, so the fractions
  // in the item mean the same thing here and in the render. ResizeObserver is an optimisation, not
  // a requirement: jsdom does not have it, and a preview that throws is worse than one that only
  // measures once.
  useEffect(() => {
    const el = box.current
    if (!el) return
    const measure = () => setBoxH(el.clientHeight)
    measure()
    if (typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    return () => ro.disconnect()
  }, [videoUrl])

  const patch = (i: number, p: Partial<TextItem>) =>
    onChange(items.map((t, k) => (k === i ? { ...t, ...p } : t)))

  return (
    <div className="space-y-2" data-testid={testId}>
      {videoUrl && (
        <div ref={box} className="relative overflow-hidden rounded bg-black" data-testid={`${testId}-preview`}>
          <video src={videoUrl} controls className="block max-h-64 w-full object-contain"
            onTimeUpdate={(e) => setAt(e.currentTarget.currentTime)}
            onLoadedMetadata={() => setBoxH(box.current?.clientHeight ?? 0)} />
          {items.map((t, i) => visibleAt(t, at) && (
            <div key={i} style={{ ...anchorStyle(t), ...textStyle(t, boxH) }}
              data-testid={`${testId}-preview-${i}`}>{t.text}</div>
          ))}
        </div>
      )}

      {items.length === 0 && <p className="text-xs text-neutral-500">No captions yet.</p>}

      {items.map((t, i) => (
        <div key={i} className="rounded border border-neutral-200 p-2" data-testid={`${testId}-row-${i}`}>
          <div className="flex items-start gap-2">
            <textarea value={t.text} rows={1} disabled={busy}
              aria-label={`caption ${i + 1} text`} data-testid={`${testId}-text-${i}`}
              placeholder="Открий своя аромат"
              onChange={(e) => patch(i, { text: e.target.value })}
              className="min-w-0 flex-1 rounded border border-neutral-300 px-2 py-1 text-sm" />
            <button type="button" disabled={busy} data-testid={`${testId}-remove-${i}`}
              onClick={() => onChange(items.filter((_, k) => k !== i))}
              className="rounded border border-neutral-300 bg-white px-2 py-1 text-xs text-neutral-600">
              Remove
            </button>
          </div>
          <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-neutral-600">
            <label className="flex items-center gap-1">from
              <NumField value={t.start ?? 0} min={0} disabled={busy} suffix="s"
                label={`caption ${i + 1} start`} testId={`${testId}-start-${i}`}
                onCommit={(v) => patch(i, { start: v })} />
            </label>
            <label className="flex items-center gap-1">for
              <NumField value={t.duration ?? 2} min={0.1} disabled={busy} suffix="s"
                label={`caption ${i + 1} duration`} testId={`${testId}-duration-${i}`}
                onCommit={(v) => patch(i, { duration: v })} />
            </label>
            <select value={t.effect} disabled={busy} aria-label={`caption ${i + 1} effect`}
              onChange={(e) => patch(i, { effect: e.target.value })}
              className="rounded border border-neutral-300 bg-white px-1 py-0.5">
              {fx.map((e) => <option key={e} value={e}>{e}</option>)}
            </select>
            <select value={t.align} disabled={busy} aria-label={`caption ${i + 1} position`}
              onChange={(e) => patch(i, { align: parseInt(e.target.value, 10) })}
              className="rounded border border-neutral-300 bg-white px-1 py-0.5">
              {ANCHORS.map((a) => <option key={a.value} value={a.value}>{a.label}</option>)}
            </select>
            <label className="flex items-center gap-1">size
              <NumField value={(t.size ?? 0.048) * 100} min={0.5} max={100} width="w-14" disabled={busy}
                label={`caption ${i + 1} size`} testId={`${testId}-size-${i}`}
                onCommit={(v) => patch(i, { size: v / 100 })}
                suffix={<span className="text-neutral-400" title="percent of the frame height, so 9:16 and 16:9 read the same">% of height</span>} />
            </label>
            <label className="flex items-center gap-1">colour
              <input type="color" value={`#${t.colour}`} disabled={busy}
                aria-label={`caption ${i + 1} colour`} data-testid={`${testId}-colour-${i}`}
                onChange={(e) => patch(i, { colour: e.target.value.replace('#', '').toUpperCase() })}
                className="h-6 w-8 rounded border border-neutral-300" />
            </label>
            <label className="flex items-center gap-1">
              <input type="checkbox" checked={!!t.bold} disabled={busy}
                aria-label={`caption ${i + 1} bold`}
                onChange={(e) => patch(i, { bold: e.target.checked })} />bold
            </label>
            {limit != null && (t.start ?? 0) + (t.duration ?? 0) > limit + 0.01 && (
              <span className="text-amber-700" data-testid={`${testId}-overrun-${i}`}>
                past the end ({limit.toFixed(1)}s)
              </span>
            )}
          </div>
        </div>
      ))}

      <div className="flex flex-wrap items-center gap-2">
        <Button disabled={busy} onClick={() => onChange([...items, blankText()])}>Add caption</Button>
        {presets && Object.keys(presets).map((name) => (
          <button key={name} type="button" disabled={busy} data-testid={`${testId}-preset-${name}`}
            title="a starting point — every value stays editable afterwards"
            onClick={() => onChange([...items, blankText(presets[name] as Partial<TextItem>)])}
            className="rounded border border-neutral-300 bg-white px-2 py-0.5 text-xs text-neutral-700">
            + {name}
          </button>
        ))}
      </div>
      {videoUrl && (
        <p className="text-xs text-neutral-500">
          The timing is exact. Placement is approximate — the burn-in is drawn by libass, so line
          breaks and outline weight will differ a little.
        </p>
      )}
    </div>
  )
}
