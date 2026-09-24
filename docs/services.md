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

## 3. Reading the Services panel

**Setup → Services**, beside Environment. Two cards: the GPU, then the services.

The **GPU bar** is one segment per process, widths proportional to what each holds, with the
driver's own overhead in grey at the end. Hover for pid, MB and uptime. Faded segments are fixed
costs that will never be freed. This is the thing to glance at before submitting a job.

A segment labelled **unknown — vorch will not touch it** is a process holding VRAM that the
manifest cannot name. It is shown rather than hidden, with its command line, because vorch will
refuse to free it and you may need to decide. If it is something of yours, give it a `match:` entry
(§4); if it is not, it is still your call.

The **services table** carries the vocabulary that matters:

| you see | it means |
|---|---|
| `ok` | running, answering, and the right instance |
| `warn` | running, but we could not verify *which* instance, or more than one process matches |
| `fail` | an orphan, a wrong instance, or it is not answering its health probe |
| `down` | **not running — and for an on-demand service that is correct**, not a fault |
| *started outside vorch* | you launched it. Supported; vorch will not stop it |
| `ORPHAN` | ours, but the vorch that started it is gone. It holds memory until stopped |
| `verified` | identity proved — for tts, the cloned voice read out of `/proc/<pid>/environ` |
| `unverified` | we could not read the identity. A warning, never a block |
| `WRONG INSTANCE` | proved to be something else. Results from it would be wrong |
| `warm` / `phase` badge | started on demand, not at bring-up |

The **log** link under each row opens the tail of that service's log file. When a service will not
start, that is the fastest answer — and for the narration server it is also where you find which
voice a past run actually used, because every start writes a header.

The same information without a browser:

```bash
python -m vorch status          # exit code 1 if anything is a fail
python -m vorch status --json
```


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


## 5. How arbitration decides

The question is never "how much VRAM is free" — it is "will this step fit, and if not, who has the
rest". Ask it directly:

```bash
python -m vorch vram --by-process --need 12000
```

```
GPU: used=13833 free=2001 total=16303 MB (12% free)

service        pid      MB   uptime  state
tts         251592   11268   11h40m  elastic/stoppable
gemma         8536    1960   23h00m  fixed
comfy       409909     222    5h55m  elastic/stoppable
(driver)         -     383        -  context and overhead

for a step needing 12000 MB: OK
  needs 12000 MB, 2001 free; freeing comfy, tts releases about 11490 MB
  would free, in order: comfy -> tts
```

That is a real reading of this machine, and it is the answer the old `gpu` check could not give: it
said *"only 2001 MB free; ComfyUI may still hold models"* while ComfyUI held 222 MB and a narration
server nobody had stopped held 11268.

### The ladder

1. **No `nvidia-smi`** → admitted. There is nothing to measure, so nothing to refuse. (The same
   stance `reclaim` has always taken.)
2. **Enough free already** → admitted, nothing to do.
3. **Enough free once evictable holders are freed** → admitted, with a plan.
4. **Otherwise** → refused, naming the deficit, who holds what, and the command that would free it.

### Eviction order, and why

Cheapest first, always:

| class | cost to free | how |
|---|---|---|
| `elastic` | ~3 s | `POST /free` — ComfyUI drops its models **without stopping** |
| `exclusive` / `warm` | ~90 s | stop the process; it must be reloaded next time |
| `resident` | never | a fixed cost, by design |

So ComfyUI is always asked to let go before anything is killed. Freeing 222 MB rarely settles it,
but it is nearly free to try, and the order means a 90-second reload is only paid when it is
genuinely required.

**`resident` is not a blocker to remove.** Gemma is the evaluator: stopping it would break every
verdict. It is reported as *"gemma (pid 8536, 1960 MB, up 23h00m) is a fixed cost and stays"*, and
you will never see vorch suggest stopping it. Note also that its footprint grows a little with use
(the KV cache), so the manifest's `holds_mb` is a floor rather than an exact figure.

**A process the manifest cannot name is never touched.** It appears in the table as
`UNKNOWN — vorch will not touch this`, with its command line so you can decide. Refusing to guess
is deliberate: the failure mode has to be "refuse", never "kill the wrong thing".

### Reading a refusal

```
needs 11500 MB, 2001 free; freeing comfy would add about 222 MB; still 9277 MB short;
held by tts (pid 251592, 11268 MB, up 11h40m);
gemma (pid 8536, 1960 MB, up 23h00m) is a fixed cost and stays; stop it with: vorch down tts
```

The arithmetic is meant to add up as you read it: 2001 free plus 222 reclaimable is 2223, against
11500 needed, which is 9277 short. Anything holding memory that the plan will *not* free is named —
including a service excluded from eviction because we were about to use it — precisely so the
numbers reconcile. If they ever do not, the attribution is wrong; see §4 on `match`.


## 6. The warm narration server — *planned*

## 7. Diagnosis by symptom

The section to use under pressure. Each entry names the command and what you should see.

### "Only N MB free" / a render fails to allocate

```bash
python -m vorch vram --by-process
```

Read the holder table. The usual culprit is a narration server left running from an earlier
session — it is 11 GB, and on a 16 GB card nothing else fits beside it. Release it:

```bash
python -m vorch down tts
```

If the largest holder is `UNKNOWN`, vorch will not touch it and neither will it guess: the command
line is printed so you can decide.

### "The narration is in the wrong voice"

Look at the `service:tts` row in `vorch status` or Setup → Services.

- **`verified`** with the voice named — the server is reading what you think it is. The problem is
  elsewhere (wording, stress, or a stale wav: see the narration docs).
- **`WRONG INSTANCE`** — it is reading a different reference. This is the real bug, and it happens
  because the voice reaches OmniVoice *only* as `OMNIVOICE_BG_REF_AUDIO`; `serve_bg.sh`'s built-in
  default is a different announcer, and `GET /health` does not report the voice back at all, so
  nothing downstream can notice. Stop it and let vorch start it with the right reference:
  `vorch down tts`.
- **`unverified`** — we could not read `/proc/<pid>/environ` (another user, a hardened kernel). Not
  a block, by design: a server you started by hand must not be refused merely because we cannot
  inspect it. Check it yourself:

```bash
tr '\0' '\n' < /proc/<pid>/environ | grep OMNIVOICE_BG_REF
```

Any line already spoken keeps the voice it was spoken in — re-speak the affected scenes after
fixing the server.

### "A service will not start"

```bash
python -m vorch logs <name>
```

A failed start already carries the last lines of that log in its error message, so you usually do
not have to go looking. If the log does not exist at all, vorch never started that service — which
for something you launch by hand is expected, and its output is in your own terminal.

### "It is running, but vorch says it is down"

The `match:` rules cannot find it. See §4 — this is nearly always a launcher that execs or forks
into something with a different command line.

### "An orphan is holding the card"

`SIGKILL` cannot be caught, so a hard-killed `vorch serve` leaves its children running. The pidfile
in `~/vorch/run/` is what makes them findable afterwards: it records the pid *and* the `owner_pid`
of the vorch that started it, so an orphan is exactly "a pidfile whose owner is gone". It shows as
a **fail** with the memory it holds. `vorch down <name>` releases it.

### "Submit is blocked"

Only two new things block a submit, and both are genuine defects rather than untidiness: an orphan
holding VRAM, and a service proved to be the wrong instance. A service simply not being managed by
vorch never blocks anything. The wizard's `force` remains for the cases where you know better.

### "Verdicts are nonsense" / "I can't see the image"

Not a service-lifecycle problem: llama-server was started without `--mmproj`. Invariant #5;
`vorch setup check --deep` has a vision probe that catches it.

## 8. Logs

```
~/vorch/logs/vorch.log        vorch's own output (CLI and `vorch serve` alike)
~/vorch/logs/vorch.log.1..5   rotated at 8 MiB
~/vorch/logs/<service>.log    one per supervised service — its stdout AND stderr
~/vorch/run/<service>.json    pidfile: pid, argv, owner_pid, identity, last use
```

`vorch.log` is configured in one place, so it exists whether you came in through the CLI or through
`vorch serve`. `--log-level debug` turns up the detail; `--log-file ''` turns the file off and keeps
stderr only.

A **service's** log is its own output, redirected to an appended file descriptor — never to
`DEVNULL`, and never to a pipe that is only read when something goes wrong, which is how OmniVoice's
diagnostics used to be thrown away entirely. Because it is a plain fd, the output survives even a
`SIGKILL` of vorch.

Every start writes a header line: the resolved argv, the cwd, and the identity environment values.
That header is why *"which voice was this server reading?"* is answerable from the log weeks later.

Rotation happens **at start**, not continuously — you cannot rotate a file descriptor another
process is holding open. A service that runs for weeks and logs heavily is therefore unbounded
until it is restarted; `vorch status` shows log sizes so it stays visible.

```bash
python -m vorch logs tts -n 50
```

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
