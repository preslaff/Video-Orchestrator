-- 0002: setup phase — projects, output profiles, job <-> project link, job kind.

CREATE TABLE projects (
    id          text PRIMARY KEY,
    slug        text NOT NULL UNIQUE,
    name        text NOT NULL,
    created_at  timestamptz NOT NULL,
    updated_at  timestamptz NOT NULL,
    data        jsonb NOT NULL              -- full Project (assets inline)
);

CREATE TABLE profiles (
    id          text PRIMARY KEY,
    name        text NOT NULL,
    kind        text NOT NULL,
    purpose     text NOT NULL,
    created_at  timestamptz NOT NULL,
    updated_at  timestamptz NOT NULL,
    data        jsonb NOT NULL              -- full OutputProfile
);

ALTER TABLE jobs ADD COLUMN project_id text REFERENCES projects (id) ON DELETE SET NULL;
ALTER TABLE jobs ADD COLUMN kind text NOT NULL DEFAULT 'video';
CREATE INDEX idx_jobs_project ON jobs (project_id, updated_at DESC);
CREATE INDEX idx_jobs_profile ON jobs ((data->>'profile_id'));
