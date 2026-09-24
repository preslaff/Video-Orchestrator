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

## 4. The manifest — *planned*

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
