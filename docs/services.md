# The services vorch depends on — what they cost and who starts them

> **Status (2026-09-24):** sections 1–2 and 9 describe how things work **today**, with every
> service started by hand. The remaining sections (`vorch up`, the manifest, arbitration, the
> Services panel, diagnosis by symptom, logs) arrive with the service-manager stages and are
> marked *planned* below. Nothing here is aspirational: the VRAM figures are measured, not
> estimated.

vorch is a deterministic Python core that drives other people's servers over HTTP. It does not
contain a model. That makes the list of things that must be running before it can do anything
longer than it looks, and the 16 GB card means two of them cannot be resident at once.

## 1. The six services

| service | port | what it does | who starts it | VRAM (measured) |
|---|---|---|---|---|
| **postgres** | 5432 | all job state, decisions, profiles, projects | systemd, once | — |
| **gemma** (llama-server) | 8080 | storyboards, prompts, and every vision verdict | you | **1838 MiB** |
| **comfy** (ComfyUI) | 8188 | every image and video actually generated | you | **222 idle**, ~22000 with Wan loaded |
| **tts** (OmniVoice-BG) | 8002 | Bulgarian narration in the cloned voice | vorch, for narration | **11268 MiB** |
| **api** (`vorch serve`) | 8000 | the HTTP API + SSE the UI talks to | you | — |
| **ui** (Next.js) | 3000 | the control panel | you | — |

The card is **16303 MiB total**. Two numbers in that table decide the whole architecture:

```
  gemma      1838   resident, and yes — that is WITH --n-gpu-layers 0
+ tts       11268
+ comfy       222   idle
= 13328        →   2975 MiB left, and Wan 2.2 wants ~22000 with RAM offload
```

**OmniVoice and a video render do not fit on this card together.** Not "tightly" — not at all.
So narration is *arbitrated*, not scheduled: the narration server exists for as long as it is
being used and no longer, and something has to make sure it is gone before Wan loads.

Two facts worth having in writing, because both contradict what the docs used to say:

- **`--n-gpu-layers 0` is not free.** Gemma pinned to the CPU still holds 1838 MiB of the card
  for its CUDA context and projector scratch. Invariant #1's "Gemma on CPU so ComfyUI owns the
  GPU" is true about *compute*, not about memory. Budget the 1.8 GB.
- **OmniVoice wants 11268 MiB, not ~5.3 GB.** The 5.3 GB figure that appeared in
  `clients/tts.py`, `config.py`, `orchestrator.py` and `queue.py` was wrong by more than 2×.
  It is why "it fits beside a loaded ComfyUI" read as merely tight rather than impossible.

### Starting them by hand

This is the normal way to work today, and it stays supported forever — see §9. The commands
live in **[README.md → Running the services](../README.md#running-the-services)**, which is the
one canonical copy. If you find a fourth copy of the llama-server line somewhere, it is stale;
delete it and point at the README.

The only ordering that actually matters: **postgres before `vorch serve`** (it opens a pool at
startup), and **`vorch serve` before the UI** (the UI proxies `/api` to it). Gemma and ComfyUI can
come up in any order, at any time — vorch discovers them per request, and `vorch setup check`
tells you which are missing.

## 2. Daily use — *planned*

`vorch up` / `down` / `status` / `logs`. Until that lands, use `python -m vorch setup check
--deep`, which probes every service and prints one row each.

## 2a. Starting services by hand — *planned section, permanent guarantee*

See §9. The short version: nothing vorch gains will take away your ability to launch any of
these in a terminal.

## 3. Reading the Services panel — *planned*

## 4. The manifest

The list of services is data, not code: `vorch/services_builtin/services.yaml` ships the defaults
and `~/vorch/services.yaml` (`VORCH_SERVICES_FILE`) overrides them. Read what vorch currently
believes with:

```bash
python -m vorch services list          # human, with the command for each service
python -m vorch services list --json   # resolved argv, env, matchers, health URLs
```

**Overrides merge, they do not replace.** A user entry is merged over the built-in one key by key,
one nested mapping deep, so changing one field never means restating an entry:

```yaml
# ~/vorch/services.yaml — everything else about tts keeps its built-in value
services:
  tts:
    idle_timeout: 120          # stop the narration server after 2 minutes idle, not 10
```

Two conveniences in values:

- **`${VORCH_X}`** resolves from the environment and falls back to `config.X`. So the manifest
  follows whatever you already set in `~/vorch/vorch.env` and holds no second copy of a URL. A
  name without the `VORCH_` prefix resolves from the environment only, and an unresolvable one is
  an error naming the variable.
- **`@file:PATH`** inlines a file's stripped contents, resolved relative to the service's `cwd`.
  Used for OmniVoice's reference transcript, which has to be passed verbatim. It is read late, so
  a missing file is a clear start-time error rather than an import-time crash.

### The fields that carry weight

| field | values | what it decides |
|---|---|---|
| `manage` | `never` · `adopt` · `full` | `never` = probe and print `hint` (postgres). `adopt` = report, never spawn (the UI). `full` = vorch may start and stop it. |
| `vram.class` | `none` · `resident` · `elastic` · `exclusive` | `resident` is a fixed tax that is never evictable (gemma). `elastic` can be freed *without stopping the process*, via `free_endpoint` (comfy's `/free`). `exclusive` means its presence forbids any other heavy step (tts). |
| `vram.holds_mb` | a **measured** number | What it actually costs at rest. The manifest is where measurements live, which is why the docstrings no longer carry them. |
| `vram.needs_mb` | a number | What it wants when fully loaded — what admission is checked against. |
| `lifetime` | `long` · `warm` · `phase` | `warm` is kept alive for `idle_timeout` and then reaped, but only ever a server vorch itself started. `warm` without an `idle_timeout` is rejected, because that is just "left running" — the bug, not the feature. |
| `adopt.verify` | `port` · `argv` · `env` | How hard we check that a running instance is the one we wanted. `env` reads `/proc/<pid>/environ`. |
| `adopt.on_unverified` | `warn` · `refuse` · `adopt` | What to do when identity cannot be read. Defaults to `warn` so a hand-started server is never blocked by our inability to inspect it. |
| `evict_adopted` | bool | May the arbiter stop this when it was started outside vorch? `false` makes it refuse the render instead and tell you which command frees the card. |
| `match` | `cmdline_contains`, `cwd` | How to recognise the process. |

### Why `match` exists, and when you need it

A launcher usually does not survive its own launch. `serve_bg.sh` **execs** into
`python -m omnivoice.cli.serve`; `comfy launch` **forks** `python main.py` and that *grandchild*
is what holds the VRAM; `npm run dev` becomes `next-server`. In every one of those cases nothing
of the original `argv` is left in the process that is actually running, so the derived fallback
(the executable's own name plus the `cwd`) cannot find it and the built-ins pin `match:`
explicitly. Gemma is a subtler case: its `argv` ends in `--port 8080`, so a naive "last token"
needle would be `8080`.

**If a service is running but vorch reports it as down, `match` is the field to look at.** Compare
what the manifest expects against reality:

```bash
python -m vorch services list --json | grep -A3 '"match"'
tr '\0' ' ' < /proc/<pid>/cmdline; readlink /proc/<pid>/cwd
```

### Worked example: a second voice

Point `tts` at a different reference clip and transcript. Because `${VORCH_*}` falls back to
config, the tidiest way is `~/vorch/vorch.env`:

```bash
# ~/vorch/vorch.env
VORCH_TTS_REF_AUDIO=Georgi/14s.wav
VORCH_TTS_REF_TEXT_FILE=Georgi/14s.txt
```

Nothing in `services.yaml` needs to change, and `vorch services list` will show the new values —
which is also how you confirm the transcript was found before a run depends on it.

To move ComfyUI to another checkout, override the paths instead:

```yaml
services:
  comfy:
    cwd: ~/comfy/ComfyUI-test
    argv: [~/comfy/ComfyUI-test/venv/bin/python, main.py, --port, "8188"]
    match: {cmdline_contains: ["main.py"], cwd: ~/comfy/ComfyUI-test}
```

> **The numbers in this file are measurements.** If you change a model, a quantisation or a
> resolution, re-measure with `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` and
> update `holds_mb` / `needs_mb`. Do not guess — a wrong figure here makes the arbiter either
> refuse work that would have fit or admit work that OOMs.


## 5. How arbitration decides — *planned*

## 6. The warm narration server — *planned*

## 7. Diagnosis by symptom — *planned*

Until it exists, the two that come up most:

- **"only N MB free" / a render OOMs.** Something is holding the card that you cannot see from
  vorch. `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` names the pid, and
  `cat /proc/<pid>/cmdline | tr '\0' ' '` names the process. The usual culprit is an OmniVoice
  server left running from an earlier narration.
- **"the narration is in the wrong voice."** The voice is passed to OmniVoice only as the
  environment variable `OMNIVOICE_BG_REF_AUDIO`, and `GET /health` does not report it — so a
  server started with `serve_bg.sh`'s *default* reference reads every line in a different
  announcer's voice and nothing notices. Check the running server's actual voice with
  `tr '\0' '\n' < /proc/<pid>/environ | grep OMNIVOICE_BG_REF_AUDIO`.

## 8. Logs — *planned*

Today there is one stream: whatever `vorch serve` prints to its terminal. The supervised
OmniVoice child's output is discarded outright (`stdout=DEVNULL`), which is the first thing the
logging stage fixes.

## 9. What vorch will not do, and why

The boundary, written down so it does not drift:

- **It will not take away hand-starting.** Launching a service in a terminal is how a voice gets
  auditioned and how anything gets debugged. Every service can be run entirely by hand,
  permanently; `vorch up` adopts whatever is already running and restarts nothing; `vorch down`
  will not stop something it did not start without `--force`; and any service can be set to
  "monitor only" for good. Configuring *every* service that way is legitimate — it gives exactly
  today's behaviour plus real diagnostics.
- **It will not restart a crashed service.** A model that OOMs will OOM again. A restart loop on
  a 16 GB card burns hours and fills the disk. vorch reports the failure with the tail of the
  service's log and stops.
- **It will not kill a process it cannot name.** If something holds the card and vorch cannot
  identify it, it refuses the render and tells you what it found. Refusing is always preferred
  over guessing.
- **It will not manage PostgreSQL or the UI.** Postgres is already a systemd unit and
  `vorch db init` creates a database inside a running server, not the server. The UI is a Node
  toolchain with hot reload and no stake in the GPU. Both are reported, never spawned.
- **It will not install anything** — not models, not ComfyUI custom nodes, not Python
  environments. It reads what is on disk and tells you when something is missing.
- **It will not become a container stack or a process supervisor.** No docker-compose, no
  supervisord, no systemd units generated per service. The reasoning is in
  `DESIGN.md` §11–12 and in the plan: a supervisor's job is to keep processes up, and vorch's
  hardest job is the opposite — stopping one to free 11 GB.
