-- 0003: decisions carry the scene's stable id.
--
-- `scene_index` is a *position*, and scenes can now be deleted, reordered and inserted, which
-- renumbers them. Old rows keep only the number they were logged with (history, read as-is);
-- new rows also carry the id, so per-scene views stay right across a reorder.

ALTER TABLE decisions ADD COLUMN scene_id text;
