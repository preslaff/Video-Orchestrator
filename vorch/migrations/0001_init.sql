-- 0001: baseline schema (jobs + decision log), ported from the SQLite era.
-- Tables are created unqualified so they land in the first schema of search_path
-- (tests run each case in its own throwaway schema).

CREATE TABLE jobs (
    id          text PRIMARY KEY,
    status      text NOT NULL,
    idea        text NOT NULL DEFAULT '',
    created_at  timestamptz NOT NULL,
    updated_at  timestamptz NOT NULL,
    data        jsonb NOT NULL              -- full Job (resumable state)
);
CREATE INDEX idx_jobs_updated ON jobs (updated_at DESC);

CREATE TABLE decisions (
    id          bigserial PRIMARY KEY,
    job_id      text NOT NULL REFERENCES jobs (id) ON DELETE CASCADE,
    scene_index integer,
    phase       text NOT NULL,
    attempt     integer NOT NULL DEFAULT 1,
    action      text NOT NULL,
    score       integer,
    reason      text NOT NULL DEFAULT '',
    created_at  timestamptz NOT NULL
);
CREATE INDEX idx_decisions_job ON decisions (job_id, id);
