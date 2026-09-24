import { describe, expect, it } from 'vitest'
import { filterAssets, matchesQuery, topTags, isImageAsset, isVideoAsset } from './assetFilter'
import type { AssetOut } from '@/api/types'

const a = (id: string, extra: Partial<AssetOut>): AssetOut => ({
  id, kind: 'reference', name: `${id}.jpg`, label: '', label_source: '', note_caption: '', note_tags: [], note_text: '',
  describe_state: '', describe_error: '', sha256: '', created_at: '', path: '', url: '', thumb_url: '', ...extra,
})

const plaque = a('p', { label: 'grey wall — a plaque', note_text: 'MIZEN HEAD VISITOR CENTRE', note_tags: ['plaque', 'interior'] })
const lamp = a('l', { note_caption: 'A Fresnel lens in the lamp room', note_tags: ['lamp_room', 'interior'] })
const tower = a('t', { label: 'rocky coastline — a white lighthouse', note_tags: ['lighthouse', 'exterior'] })

describe('asset search', () => {
  it('matches label, caption, tags and sign text, every word, any case', () => {
    expect(matchesQuery(plaque, 'mizen')).toBe(true)
    expect(matchesQuery(lamp, 'fresnel ROOM')).toBe(true)
    expect(matchesQuery(tower, 'lighthouse fog')).toBe(false)
    expect(matchesQuery(tower, '   ')).toBe(true)
    expect(matchesQuery(a('x', { name: 'IMG_0042.jpg' }), 'img_0042')).toBe(true)
  })

  it('filters by query and tags together, keeping order', () => {
    expect(filterAssets([plaque, lamp, tower], '', ['interior']).map((x) => x.id)).toEqual(['p', 'l'])
    expect(filterAssets([plaque, lamp, tower], 'lamp', ['interior']).map((x) => x.id)).toEqual(['l'])
  })

  it('ranks tags by how many assets carry them, dropping one-offs', () => {
    expect(topTags([plaque, lamp, tower, a('t2', { note_tags: ['lighthouse'] })])).toEqual(['interior', 'lighthouse'])
  })
})

it('separates clips from images, by allow-list not deny-list', () => {
  // The old rule was `kind !== 'master_face'`, which offered anything new everywhere.
  const img = { kind: 'reference' }
  const face = { kind: 'master_face' }
  const clip = { kind: 'video' }
  expect([img, face, clip].filter(isImageAsset).map((x) => x.kind)).toEqual(['reference', 'master_face'])
  expect([img, face, clip].filter(isVideoAsset).map((x) => x.kind)).toEqual(['video'])
})
