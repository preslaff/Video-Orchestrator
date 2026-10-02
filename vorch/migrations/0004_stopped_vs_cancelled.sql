-- 0004: "cancelled" becomes "stopped", and "cancelled" is freed for a real cancel.
--
-- The old status meant "stopped at a safe point, finished work kept, resume with retry" — the UI
-- already labelled it *Stopped*, and only the enum said otherwise. That mismatch was actively
-- misleading: a job one had finished with looked identical to one merely paused, and there was no
-- way to say "never run this again". So the resumable state takes the name it always had in the
-- interface, and `cancelled` now means terminal: `retry` refuses it.
--
-- Status lives in two places — the `status` column and the job's JSONB snapshot — and both have
-- to move together, or a job would load with one value and list with the other.

UPDATE jobs
   SET status = 'stopped',
       data = jsonb_set(data, '{status}', '"stopped"')
 WHERE status = 'cancelled';
