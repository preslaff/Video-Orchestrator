"""vorch CLI — run pipeline steps individually or a full single scene.

Individual step commands isolate each ComfyUI endpoint for testing; ``run-scene``
chains them. Image arguments are local paths and are staged into ComfyUI input/
automatically.

    python -m vorch keyframe "oil painting, woman kneading bread, impasto"
    python -m vorch animate keyframe.png "slow camera push-in, hands kneading"
    python -m vorch faceswap scene.png assets/jena-cropped.jpeg
    python -m vorch run-scene --keyframe-prompt "..." --animation-prompt "..." \
        --master assets/jena-cropped.jpeg
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from . import config, formats, logsetup, provenance
from .clients import comfy
from .clients.gemma import GemmaError
from .evaluator import EvaluationError, evaluate_image
from .media import MediaError
from .orchestrator import GATE_KEYFRAMES, GATE_STORYBOARD, Orchestrator
from .pipeline import Scene, run_scene, run_sequence
from .db import DatabaseUnavailable
from .genparams import DEFAULT, GenParams
from .state import JobStatus, Store
from .steps.animate import animate
from .steps.faceswap import faceswap
from .steps.keyframe import keyframe
from .storyboard import StoryboardError

GATES = {GATE_STORYBOARD, GATE_KEYFRAMES}


def _params_from_args(args) -> GenParams:
    """GenParams for the scene/sequence commands: --aspect picks sizes, --length/--fps override."""
    aspect = getattr(args, "aspect", None) or DEFAULT.aspect
    overrides = {}
    if getattr(args, "length", None):
        overrides["length"] = args.length
    if getattr(args, "fps", None):
        overrides["fps"] = args.fps
    return GenParams.for_aspect(aspect, **overrides)


def _cmd_keyframe(args) -> int:
    width, height = args.width, args.height
    if args.aspect:
        width, height = formats.keyframe_size(args.aspect)
    path = keyframe(
        args.prompt,
        workflow=args.workflow,
        width=width,
        height=height,
        steps=args.steps,
        cfg=args.cfg,
        seed=args.seed,
        unet=args.unet,
        negative=args.negative,
    )
    print(path)
    return 0


def _cmd_animate(args) -> int:
    name = comfy.upload_image(args.image)
    width, height = args.width, args.height
    if args.aspect:
        width, height = formats.wan_size(args.aspect)
    kw = {}
    if args.negative is not None:
        kw["negative"] = args.negative
    path = animate(
        name, args.prompt, workflow=args.workflow, width=width, height=height,
        length=args.length, fps=args.fps, seed=args.seed, **kw,
    )
    print(path)
    return 0


def _cmd_edit(args) -> int:
    from .pipeline import stage_references
    from .steps.edit import edit

    params = GenParams.for_aspect(args.aspect) if args.aspect else DEFAULT
    names = stage_references(args.ref, params, fit=not args.no_fit)
    kw = params.edit_kwargs()
    kw["workflow"] = args.workflow
    if args.negative is not None:
        kw["negative"] = args.negative
    path = edit(args.prompt, names, seed=args.seed, **kw)
    print(path)
    return 0


def _cmd_faceswap(args) -> int:
    input_name = comfy.upload_image(args.input_image)
    source_name = comfy.upload_image(args.source_image)
    path = faceswap(input_name, source_name, workflow=args.workflow, restore_model=args.restore_model)
    print(path)
    return 0


def _cmd_evaluate(args) -> int:
    verdict = evaluate_image(args.image, args.task)
    print(verdict.model_dump_json(indent=2))
    print(f"decision: {verdict.decision}", file=sys.stderr)
    if verdict.needs_review:
        print(f"flag: accepted but score={verdict.score} — review", file=sys.stderr)
    return 0 if verdict.decision == "accept" else 2


def _cmd_run_scene(args) -> int:
    if not args.no_face_swap and not args.master:
        raise ValueError("--master is required unless --no-face-swap is given")
    result = run_scene(
        args.keyframe_prompt,
        args.animation_prompt,
        args.master or "",
        seed=args.seed,
        evaluate=args.evaluate,
        task=args.task,
        max_attempts=args.max_attempts,
        face_swap=not args.no_face_swap,
        params=_params_from_args(args),
    )
    print(f"keyframe: {result.keyframe}")
    print(f"swapped:  {result.swapped}")
    print(f"video:    {result.video}")
    if result.keyframe_verdict is not None:
        v = result.keyframe_verdict
        print(f"verdict:  {v.verdict} (score={v.score})")
    return 0


def _load_scenes(path: str) -> list[Scene]:
    """Parse a YAML scenes file into Scene objects.

    Expected shape: a list of mappings, each with ``keyframe_prompt`` and
    ``animation_prompt`` (required) and optional ``continuous`` (bool).
    """
    import yaml  # local import: pyyaml is only needed for run-sequence

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path}: expected a non-empty list of scenes")
    scenes: list[Scene] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: scene {i} is not a mapping")
        try:
            scenes.append(
                Scene(
                    keyframe_prompt=item["keyframe_prompt"],
                    animation_prompt=item["animation_prompt"],
                    continuous=bool(item.get("continuous", False)),
                )
            )
        except KeyError as e:
            raise ValueError(f"{path}: scene {i} missing required field {e}") from e
    return scenes


def _cmd_run_sequence(args) -> int:
    if not args.no_face_swap and not args.master:
        raise ValueError("--master is required unless --no-face-swap is given")
    scenes = _load_scenes(args.scenes)
    results = run_sequence(
        scenes,
        args.master or "",
        seed=args.seed,
        evaluate=args.evaluate,
        relock_identity=not args.no_relock,
        max_attempts=args.max_attempts,
        face_swap=not args.no_face_swap,
        params=_params_from_args(args),
    )
    for i, r in enumerate(results):
        tag = "continuous" if r.continued_from_previous else "fresh"
        print(f"scene {i + 1} [{tag}]")
        print(f"  start:  {r.start_image}")
        print(f"  video:  {r.video}")
        if r.keyframe_verdict is not None:
            v = r.keyframe_verdict
            print(f"  verdict: {v.verdict} (score={v.score})")
    return 0


def _open_store() -> Store:
    return Store.open()


def _print_job(job, store: Store) -> None:
    print(f"job {job.id}  [{job.status.value}]  {job.idea!r}")
    for s in job.scenes:
        kf = s.keyframe_path or "-"
        vid = s.video_path or "-"
        print(f"  scene {s.index} [{s.status.value}]  kf={kf}  video={vid}")
    if job.status is JobStatus.NEEDS_APPROVAL:
        gate = Orchestrator.next_gate(job, store)
        print(f"  -> paused. approve with:  python -m vorch approve {job.id} {gate}")


def _print_provenance(job) -> None:
    """Show the recorded seed + prompt for each scene's media (from sidecars)."""
    rows = []
    for s in job.scenes:
        for label, path in (("keyframe", s.keyframe_path), ("video", s.video_path)):
            if not path:
                continue
            m = provenance.read_manifest(path)
            name = Path(path).name
            if not m:
                rows.append((s.index, label, None, None, name))
                continue
            p = m.get("params", {})
            seed = p.get("seed", p.get("seed_high"))
            rows.append((s.index, label, seed, p.get("positive", ""), name))
    if not rows:
        return
    print("provenance (seed + prompt per output):")
    for idx, label, seed, prompt, name in rows:
        if seed is None and prompt is None:
            print(f"  s{idx} {label:8} {name}  (no manifest)")
            continue
        print(f"  s{idx} {label:8} seed={seed}  {name}")
        if prompt:
            print(f"           prompt: {prompt}")


def _orchestrator(store: Store, spec: dict) -> Orchestrator:
    """Orchestrator for a job.yaml: profile from `profile:` / `aspect:` / legacy knobs."""
    from . import profiles

    return Orchestrator(
        store,
        profile=profiles.from_legacy_spec(spec, store),
        n_scenes=spec.get("n_scenes"),
        seed=int(spec.get("seed", config.SEED_RANDOM)),
    )


def _export_if_done(store: Store, job, *, enabled: bool = True) -> None:
    from . import export, projects

    if not enabled or job.status is not JobStatus.DONE or job.export:
        return
    project = projects.load(store, job.project_id) if job.project_id else None
    rec = export.export_job(store, job, project)
    job.export = rec.to_dict()
    store.save_job(job)
    print(f"exported {len(rec.files)} file(s) -> {rec.path}" + ("" if rec.ok else f"  ({rec.error})"))


def _cmd_run(args) -> int:
    import yaml  # local import: pyyaml only needed for job specs

    with open(args.spec, encoding="utf-8") as f:
        spec = yaml.safe_load(f) or {}
    if not spec.get("idea"):
        raise ValueError(f"{args.spec}: 'idea' is required")

    from . import projects

    store = _open_store()
    try:
        o = _orchestrator(store, spec)
        project = None
        if spec.get("project"):
            project = projects.load_by_slug(store, spec["project"]) or projects.load(store, spec["project"])
            if project is None:
                raise ValueError(f"{args.spec}: unknown project {spec['project']!r}")
        master = spec.get("master_face", "")
        if not master and project is not None and projects.master_face(project):
            master = projects.master_face(project).path
        if o.face_swap and not master:
            raise ValueError(f"{args.spec}: 'master_face' is required when face_swap is on")
        job = o.create_job(
            spec["idea"],
            style=spec.get("style", ""),
            master_face=master,
            autonomy=spec.get("autonomy", "semi"),
            mode=spec.get("mode", "t2i"),
            reference_images=spec.get("references") or [],
            project_id=project.id if project else None,
        )
        o.run(job)
        _print_job(job, store)
        _export_if_done(store, job, enabled=bool(spec.get("export", True)))
    finally:
        store.close()
    return 0


def _cmd_approve(args) -> int:
    if args.gate not in GATES:
        raise ValueError(f"unknown gate {args.gate!r}; expected one of {sorted(GATES)}")
    store = _open_store()
    try:
        job = store.load_job(args.job_id)
        if job is None:
            raise ValueError(f"no job with id {args.job_id}")
        o = Orchestrator.from_job(store, job)  # run-config restored from the job
        o.approve(job, args.gate)
        _print_job(job, store)
        _export_if_done(store, job)
    finally:
        store.close()
    return 0


def _cmd_jobs(args) -> int:
    store = _open_store()
    try:
        rows = store.list_jobs()
        if not rows:
            print("no jobs")
        for r in rows:
            print(f"{r.id}  [{r.status.value}]  {r.updated_at}  {r.idea!r}")
    finally:
        store.close()
    return 0


def _cmd_serve(args) -> int:
    import uvicorn

    from .api import create_app

    app = create_app(_open_store())
    level = logging.getLevelName(logsetup.resolve_level(args.log_level)).lower()
    print(f"API on http://{args.host}:{args.port}/api  (docs: /docs)")
    print(f"UI: {config.UI_URL}  (run `npm run dev` in frontend/)")
    if (path := logsetup.configure(args.log_level, log_file=args.log_file)) is not None:
        print(f"log: {path}")
    uvicorn.run(app, host=args.host, port=args.port, log_level=level)
    return 0


def _cmd_openapi(args) -> int:
    from .api import openapi_schema

    json.dump(openapi_schema(), sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0




def _cmd_db(args) -> int:
    from . import db

    if args.db_command == "init":
        created = db.ensure_database(config.DATABASE_URL)
        store = Store(config.DATABASE_URL)
        try:
            print(f"database {db.dbname_of(config.DATABASE_URL)} "
                  f"{'created' if created else 'exists'}; schema version {store.version}")
        finally:
            store.close()
        return 0
    if args.db_command == "status":
        store = _open_store()
        try:
            print(f"dsn: {config.DATABASE_URL}")
            print(f"schema version: {store.schema_version()} (latest {db.latest_version()})")
            print(f"jobs: {len(store.list_jobs())}")
        finally:
            store.close()
        return 0
    if args.db_command == "import-sqlite":
        from .sqlite_import import import_sqlite

        store = _open_store()
        try:
            r = import_sqlite(args.path, store)
        finally:
            store.close()
        print(f"imported {r.jobs} jobs, {r.decisions} decisions "
              f"({r.skipped_jobs} jobs already present, skipped)")
        return 0
    return 2


def _cmd_setup(args) -> int:
    from . import envcheck, profiles

    store = _open_store()
    try:
        profile = profiles.resolve(store, args.profile) if args.profile else None
        report = envcheck.run_checks(store=store, profile=profile, deep=args.deep, force=True)
    finally:
        store.close()
    marks = {"ok": "ok  ", "warn": "WARN", "fail": "FAIL"}
    for c in report.checks:
        print(f"[{marks[c.status]}] {c.name:<24} {c.detail}")
    print(f"-> {report.status} ({report.duration_ms} ms)" + ("" if report.ok else "  — fix the FAIL rows or submit with force"))
    return 0 if report.ok else 1


def _cmd_profiles(args) -> int:
    import yaml

    from . import profiles

    store = _open_store()
    try:
        if getattr(args, "import_file", None):
            # {name, base, overrides} -> a new user profile (same shape as POST /api/profiles)
            with open(args.import_file, encoding="utf-8") as f:
                doc = yaml.safe_load(f) or {}
            if not doc.get("name"):
                raise ValueError(f"{args.import_file}: 'name' is required")
            base = profiles.resolve(store, doc.get("base") or profiles.DEFAULT_PROFILE_ID)
            p = profiles.clone(base, str(doc["name"]), overrides=doc.get("overrides") or {})
            profiles.save(store, p)
            print(f"imported profile {p.id}  ({p.name}; base {base.id})")
            return 0
        for p in profiles.list_all(store):
            size = f"{p.keyframe.width}x{p.keyframe.height}"
            if p.video:
                size += f" / {p.video.width}x{p.video.height}@{p.video.length}f {p.video.fps}fps"
            else:
                size += f" / stills x{p.stills.variants}"
            print(f"{p.id:<20} {p.kind:<7} {p.aspect:<5} {size:<40} {p.name}" + ("" if p.builtin else "  [user]"))
    finally:
        store.close()
    return 0


def _cmd_projects(args) -> int:
    from . import projects

    store = _open_store()
    try:
        rows = projects.list_all(store)
        if not rows:
            print("no projects (create one: vorch project add <name>)")
        for p in rows:
            print(f"{p.slug:<24} {p.name:<30} profile={p.default_profile_id:<16} "
                  f"assets={len(p.assets):<3} jobs={store.count_jobs(project_id=p.id)}")
    finally:
        store.close()
    return 0


def _select_for_labelling(assets, args) -> tuple[list, int]:
    """Assets to (re)label, plus how many human-authored labels were left alone.

    Default is the cheap, resumable case: only assets nobody and nothing has labelled.
    ``--force`` also redoes machine suggestions; a person's wording is replaced only
    under the explicit ``--overwrite-human``.
    """
    todo, skipped_human = [], 0
    for a in assets:
        if args.kind and a.kind != args.kind:
            continue
        if a.label_source == "human" and not args.overwrite_human:
            if a.label.strip():
                skipped_human += 1
            continue
        if args.relabel:
            if a.note:
                todo.append(a)
            continue
        if not args.force and a.label.strip():
            continue
        todo.append(a)
    if args.limit:
        todo = todo[: args.limit]
    return todo, skipped_human


def _label_assets(project, args, store) -> int:
    """Describe project assets with Gemma and propose labels for human review."""
    import time

    from . import asset_vision, projects

    todo, skipped_human = _select_for_labelling(project.assets, args)
    hint = args.hint if args.hint is not None else project.describe_hint  # the project's vocabulary by default
    if skipped_human:
        print(f"skipping {skipped_human} label(s) edited by hand (use --overwrite-human to replace)")
    if not todo:
        print("nothing to label" + ("" if args.force or args.relabel else " (all assets already have labels; --force to redo)"))
        return 0
    verb = "relabelling" if args.relabel else "describing"
    print(f"{verb} {len(todo)} asset(s)" + (" [dry run]" if args.dry_run else ""))

    started, failed = time.monotonic(), 0
    for i, a in enumerate(todo, 1):
        eta = ""
        if i > 1 and not args.relabel:
            per = (time.monotonic() - started) / (i - 1)
            eta = f"  eta {int(per * (len(todo) - i + 1)) // 60}m"
        try:
            if args.relabel:
                note = asset_vision.AssetNote.model_validate(a.note)
            else:
                note = asset_vision.describe(a.path, hint=hint, kind=a.kind)
        except Exception as e:   # one bad file must not end a 40-minute run
            failed += 1
            print(f"[{i}/{len(todo)}] {a.name}: FAILED — {e}")
            continue
        label = asset_vision.label_for(note, kind=a.kind, fallback=Path(a.name).stem)
        print(f"[{i}/{len(todo)}]{eta}  {a.name}\n    {a.label or '(no label)'}\n  → {label}")
        if a.kind == "master_face" and not note.is_portrait:
            print("    warning: no clear single face — this may be the wrong file for an identity")
        if not args.dry_run:
            projects.set_asset_note(project, a.id, note.model_dump(), label=label)
            projects.save(store, project)   # per asset: Ctrl-C loses at most one
    if args.dry_run:
        print("dry run: nothing written")
    else:
        print(f"labelled {len(todo) - failed} asset(s); review them in the UI or with `vorch project show {project.slug}`")
    return 1 if failed else 0


def _cmd_project(args) -> int:
    from . import profiles, projects

    store = _open_store()
    try:
        if args.project_command == "add":
            profiles.resolve(store, args.profile)  # KeyError -> ValueError below
            p = projects.new_project(args.name, default_profile_id=args.profile, export_root=args.export_root or "")
            projects.save(store, p)
            print(f"created project {p.slug} ({p.id})")
            return 0
        p = projects.load_by_slug(store, args.slug) or projects.load(store, args.slug)
        if p is None:
            raise ValueError(f"unknown project {args.slug!r}")
        if args.project_command == "add-asset":
            a = projects.add_asset(p, args.path, kind=args.kind, name=Path(args.path).name, label=args.label or "")
            projects.save(store, p)
            print(f"added {a.kind} asset {a.id}: {a.path}")
            return 0
        if args.project_command == "label-assets":
            return _label_assets(p, args, store)
        if args.project_command == "set-kind":
            a = projects.set_asset_kind(p, args.asset_id, args.kind)  # KeyError -> ValueError below
            projects.save(store, p)
            print(f"{a.id} is now {a.kind}: {a.label or a.name}")
            return 0
        if args.project_command == "show":
            print(f"{p.name} [{p.slug}]  profile={p.default_profile_id}  export={projects.export_root(p)}")
            for a in p.assets:
                mark = {"auto": " [AI]", "human": ""}.get(a.label_source, "")
                print(f"  {a.kind:<12} {a.id}  {a.name}  {a.label}{mark}")
            for s in store.list_jobs(project_id=p.id):
                print(f"  job {s.id[:8]}  [{s.status.value}]  {s.idea!r}")
            return 0
    except KeyError as e:
        raise ValueError(str(e).strip("'")) from e
    finally:
        store.close()
    return 2


def _cmd_export(args) -> int:
    from . import export, projects

    store = _open_store()
    try:
        job = store.load_job(args.job_id)
        if job is None:
            raise ValueError(f"no job with id {args.job_id}")
        if job.status is not JobStatus.DONE and not args.force:
            raise ValueError(f"job is {job.status.value}, not done (use --force)")
        project = projects.load(store, job.project_id) if job.project_id else None
        rec = export.export_job(store, job, project)
        job.export = rec.to_dict()
        store.save_job(job)
        print(f"{rec.path}")
        for f in rec.files:
            print(f"  {f}")
        for m in rec.missing:
            print(f"  MISSING {m}")
        return 0 if rec.ok else 1
    finally:
        store.close()


def _cmd_assemble(args) -> int:
    """Join a finished job's clips into one video with transitions (ffmpeg xfade)."""
    from . import assembly, export, projects

    store = _open_store()
    try:
        job = store.load_job(args.job_id)
        if job is None:
            raise ValueError(f"no job with id {args.job_id}")
        if args.list:
            cur = int((job.assembly or {}).get("version", 0) or 0)
            for v in assembly.versions_of(job):
                cuts = ", ".join(c["transition"] if c["transition"] == "none" else f"{c['transition']} {c['duration']:g}s"
                                 for c in (v.get("spec") or {}).get("cuts", []))
                spec = v.get("spec") or {}
                speeds = ", ".join(
                    f"#{i + 1} s{int(it.get('scene_index', 0)) + 1:02d} {it.get('speed', 1):g}x"
                    for i, it in enumerate(spec.get("items", [])) if it.get("target_seconds") is not None)
                extra = (f"  speed: {speeds}" if speeds else "") + ("  smooth" if spec.get("smooth") else "")
                mark = "*" if int(v.get("version", 0)) == cur else " "
                print(f"{mark} v{v['version']:<3} {v.get('duration')}s  {v.get('n_clips')} clip(s)  {cuts or 'single clip'}{extra}  {v.get('path')}")
            if not assembly.versions_of(job):
                print("no assemblies yet")
            return 0
        if args.select:
            assembly.select_version(job, args.select)
            store.save_job(job)
            print(f"v{args.select} is now current: {job.assembly['path']}")
            _export_if_done(store, job, enabled=True) if not job.export else None
            return 0
        if job.status is not JobStatus.DONE and not args.force:
            raise ValueError(f"job is {job.status.value}, not done (use --force)")
        if args.spec:
            with open(args.spec, encoding="utf-8") as f:
                spec_dict = json.load(f)
        else:
            n = len(assembly.clips_of(job))
            spec_dict = {"cuts": [{"transition": args.transition, "duration": args.duration} for _ in range(max(0, n - 1))]}
        if args.encoder:
            spec_dict["encoder"] = args.encoder
        if args.fps:
            spec_dict["fps"] = args.fps
        if args.delivery:
            spec_dict["delivery"] = args.delivery
        if args.width:
            spec_dict["width"] = args.width
        if args.height:
            spec_dict["height"] = args.height
        if args.grain:
            spec_dict["grain"] = args.grain
        spec, paths = assembly.resolve(job, spec_dict)
        print(f"assembling {len(paths)} clip(s): " + ", ".join(
            c.transition if c.transition == "none" else f"{c.transition} {c.duration:g}s" for c in spec.cuts))
        rec = assembly.run(job, spec, paths)
        assembly.register_version(job, rec)
        store.save_job(job)
        if not rec.ok:
            print(f"error: {rec.error}", file=sys.stderr)
            return 1
        src = assembly.profile_video(job)
        upscaled = src.get("width") and rec.width and rec.width > int(src["width"])
        note = f"  (upscaled from {src['width']}x{src['height']})" if upscaled else ""
        print(f"v{rec.version}: {rec.path}  ({rec.duration}s, {rec.width}x{rec.height} @ {rec.fps:g} fps){note}")
        if job.export or job.status is JobStatus.DONE:
            project = projects.load(store, job.project_id) if job.project_id else None
            er = export.export_job(store, job, project)
            job.export = er.to_dict()
            store.save_job(job)
            print(f"exported -> {er.path}")
        return 0
    finally:
        store.close()


def _cmd_up(args) -> int:
    """Bring the world up in dependency order. Additive: adopts whatever is already running."""
    from . import services, supervisor, vram

    specs = services.load(force=True)
    if args.dry_run:
        print(f"start order: {' -> '.join(services.order(specs, args.names or None))}")
        for name in services.order(specs, args.names or None):
            s = specs[name]
            state = supervisor.resolve(s)
            if state.running:
                verdict = "already running — would adopt it"
            elif s.manage != "full":
                verdict = f"not vorch's to start ({s.manage}) — {s.hint or 'start it yourself'}"
            elif s.lifetime in ("warm", "phase") and name not in (args.names or ()):
                verdict = f"{s.lifetime}; started on demand, not here"
            else:
                verdict = f"would start: cd {s.cwd_path} && {' '.join(s.argv_expanded())}"
            print(f"  {name:<9} {verdict}")
        return 0

    failed: list[str] = []
    running: list[str] = []
    for name in services.order(specs, args.names or None):
        spec = specs[name]
        before = supervisor.resolve(spec)
        if before.orphan and not args.reclaim_orphans:
            print(f"[FAIL] {name:<9} orphan from an earlier vorch (pid {before.pid}"
                  f"{f', {before.vram_mb} MB' if before.vram_mb else ''}) — "
                  f"rerun with --reclaim-orphans, or `vorch down {name} --force`")
            failed.append(name)
            continue
        if before.running and not before.orphan:
            extra = "adopted, started outside vorch" if before.adopted else "already running"
            print(f"[ ok ] {name:<9} {extra}"
                  f"{f'; {before.identity_detail}' if before.identity == 'verified' else ''}")
            running.append(name)
            continue
        if spec.manage != "full":
            print(f"[hint] {name:<9} {spec.hint or 'start it yourself'}")
            continue
        if spec.lifetime in ("warm", "phase") and name not in (args.names or ()):
            print(f"[skip] {name:<9} {spec.lifetime}; started on demand")
            continue
        t0 = time.monotonic()
        try:
            state = supervisor.up([name], wait=not args.no_wait,
                                  reclaim_orphans=args.reclaim_orphans)[-1]
        except (supervisor.SupervisorError, services.ServiceError, vram.VramUnavailable) as e:
            print(f"[FAIL] {name:<9} {e}")
            failed.append(name)
            continue
        took = f"{time.monotonic() - t0:.0f}s"
        held = f", {state.vram_mb} MB" if state.vram_mb else ""
        print(f"[start] {name:<8} pid {state.pid}, ready in {took}{held}  -> {spec.log_path}")
        running.append(name)

    summary = f"{len(running)} of {len(services.order(specs, args.names or None))} services up"
    if (gpu := vram.snapshot()) is not None:
        summary += f" — {gpu['free']}/{gpu['total']} MB free"
    print("\n" + summary)
    return 1 if failed else 0


def _cmd_down(args) -> int:
    from . import services, supervisor, vram

    before = vram.gpu_free_mb()
    try:
        stopped = supervisor.down(args.names or None, phase_only=args.phase_only, force=args.force)
    except services.ServiceError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if not stopped:
        print("nothing stopped (vorch only stops what it started; --force overrides)")
        return 0
    print(f"stopped: {', '.join(stopped)}")
    after = vram.gpu_free_mb()
    if before is not None and after is not None and after > before:
        print(f"VRAM freed: {before} -> {after} MB")
    return 0


def _cmd_status(args) -> int:
    """Every service, whether it is up, whether it is ours, and who holds the card."""
    from . import supervisor, vram

    table = supervisor.states(do_probe=not args.no_probe)
    if args.json:
        json.dump(supervisor.summary(table), sys.stdout, indent=2, ensure_ascii=False, default=str)
        sys.stdout.write("\n")
        return 0
    mark = {"ok": "[ ok ]", "warn": "[warn]", "fail": "[FAIL]", "down": "[down]"}
    for s in table:
        print(f"{mark[s.status]} {s.name:<9} {s.detail}")
        if s.status == "fail" and s.orphan:
            print(f"         reclaim it with: vorch down {s.name}")
    gpu = vram.attribute()
    if gpu is not None:
        named = ", ".join(f"{h.label} {h.used_mb}" for h in gpu["holders"]) or "nothing"
        print(f"\nGPU: {gpu['free']} / {gpu['total']} MB free — held by {named} MB")
    leaked = [s for s in table if s.orphan and s.spec.lifetime == "phase"]
    unattended = [s for s in table if s.orphan and s.vram_mb and s not in leaked]
    if leaked:
        print(f"\nleaked: {', '.join(s.name for s in leaked)} — `vorch down` releases them")
    if unattended:
        total = sum(s.vram_mb or 0 for s in unattended)
        print(f"\n{', '.join(s.name for s in unattended)} was started by vorch and nothing is "
              f"watching it now ({total} MB). It is still yours: `vorch down` frees it, and any "
              f"render will evict it first.")
    return 1 if any(s.status == "fail" for s in table) else 0


def _cmd_logs(args) -> int:
    from . import supervisor

    print(supervisor.tail(args.name, args.lines))
    return 0


def _cmd_services(args) -> int:
    """The manifest as parsed, with the command you could run by hand."""
    from . import services

    specs = services.load(force=True)
    if args.services_command == "list":
        if args.json:
            json.dump({"services_file": str(config.SERVICES_FILE),
                       "order": services.order(specs),
                       "services": [s.to_dict(resolve=True) for s in specs.values()]},
                      sys.stdout, indent=2, ensure_ascii=False)
            sys.stdout.write("\n")
            return 0
        print(f"manifest: {services.BUILTIN_FILE}")
        print(f"user overrides: {config.SERVICES_FILE}"
              f"{'' if config.SERVICES_FILE.is_file() else '  (absent — built-ins only)'}")
        print(f"start order: {' -> '.join(services.order(specs))}\n")
        for name in services.order(specs):
            s = specs[name]
            where = "builtin" if s.builtin else "overridden"
            vram = f"{s.vram.holds_mb} MB" if s.vram.holds_mb else "-"
            needs = f", wants {s.vram.needs_mb}" if s.vram.needs_mb else ""
            print(f"{name:<9} {s.manage:<6} {s.lifetime:<6} vram {s.vram.kind:<9} "
                  f"{vram:>9}{needs}  [{where}]")
            print(f"          {s.description}")
            if s.manage == "full":
                # Printed so it can be copied: hand-starting stays a first-class way to work.
                print(f"          run: cd {s.cwd_path} && {' '.join(s.argv_expanded())}")
                try:
                    for k, v in s.env_resolved().items():
                        shown = v if len(v) <= 60 else v[:57] + "..."
                        print(f"          env: {k}={shown}")
                except services.ServiceError as e:
                    print(f"          env: UNRESOLVED — {e}")
            elif s.hint:
                print(f"          hint: {s.hint}")
            if url := s.health_url():
                print(f"          health: GET {url}")
            elif target := s.health_target():
                print(f"          health: connect {target}")
        return 0
    raise ValueError(f"unknown services command {args.services_command!r}")


def _cmd_registry(args) -> int:
    import yaml

    from . import registry, registry_tools

    reg = registry.load_registry(force=True)
    if args.registry_command == "list":
        for spec in reg.list(args.kind):
            where = "builtin" if spec.builtin else f"user ({spec.source})"
            print(f"{spec.name:<16} {spec.kind:<12} {spec.template.name:<44} "
                  f"in={','.join(spec.inputs) or '-':<28} out={spec.output_type or '?'}  [{where}]")
        for name, err in reg.errors.items():
            print(f"{name:<16} INVALID      {err.splitlines()[0]}")
        return 0
    if args.registry_command == "validate":
        from . import backends

        known = backends.load(force=True)
        specs = [reg.get(args.name)] if args.name else reg.list()
        bad = 0
        infos: dict[str, dict | None] = {}
        for spec in specs:
            info = None
            if args.online:
                # each workflow is checked against *its* backend (or --backend)
                bname = args.backend or spec.backend
                if bname not in infos:
                    infos[bname] = comfy.object_info(backend=bname)
                info = infos[bname]
            problems = reg.validate(spec, object_info=info, backends=known)
            status = "ok" if not problems else "PROBLEMS"
            print(f"{spec.name:<16} {status}")
            for pr in problems:
                print(f"    - {pr}")
            bad += bool(problems)
        for name, err in reg.errors.items():
            print(f"{name:<16} INVALID\n    - {err}")
            bad += 1
        return 1 if bad else 0
    wf = registry.load_template_file(Path(args.file))
    if "nodes" in wf:
        print("error: this is a UI export; use 'Export (API)' in ComfyUI", file=sys.stderr)
        return 1
    if args.registry_command == "inspect":
        info = comfy.object_info(backend=args.backend) if args.online else None
        print(f"guessed kind: {registry_tools.guess_kind(wf)}")
        print(registry_tools.format_inspect(registry_tools.inspect_template(wf, info)))
        return 0
    if args.registry_command == "add":
        src = Path(args.file)
        out_dir = Path(args.out) if args.out else config.USER_WORKFLOW_DIR
        name = args.name or src.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        yaml_path = out_dir / f"{name}.yaml"
        tpl_path = out_dir / src.name
        if (yaml_path.exists() or (tpl_path.exists() and tpl_path.resolve() != src.resolve())) and not args.force:
            print(f"error: {yaml_path} or {tpl_path} exists (use --force)", file=sys.stderr)
            return 1
        if tpl_path.resolve() != src.resolve():
            tpl_path.write_bytes(src.read_bytes())
        doc, notes = registry_tools.guess_spec(wf, name=name, kind=args.kind, template=src.name)
        text = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)
        if notes:
            text += "\n# Unmapped numeric literals (promote to params if you want to control them):\n"
            text += "".join(f"#   {n}\n" for n in notes)
        yaml_path.write_text(text, encoding="utf-8")
        print(text)
        print(f"wrote {yaml_path}\nnext: edit the names/description, then `vorch registry validate {name} --online`")
        return 0
    return 2


def _cmd_vram(args) -> int:
    from . import vram

    s = vram.snapshot()
    if s is None:
        print("GPU: unavailable (no nvidia-smi)")
    else:
        pct = 100 * s["free"] / s["total"] if s["total"] else 0
        print(f"GPU: used={s['used']} free={s['free']} total={s['total']} MB ({pct:.0f}% free)")
    if args.by_process:
        # The question the old output could not answer: not how much is gone, but who has it.
        info = vram.attribute(s)
        if info is None:
            print("per-process: unavailable (no nvidia-smi)")
        elif not info["holders"]:
            print("per-process: nothing is holding the card")
        else:
            print(f"\n{'service':<10} {'pid':>7} {'MB':>7} {'uptime':>8}  state")
            for h in info["holders"]:
                state = "elastic/stoppable" if h.evictable else "fixed"
                if h.service is None:
                    state = "UNKNOWN — vorch will not touch this"
                print(f"{h.label:<10} {h.pid:>7} {h.used_mb:>7} {h.uptime_str:>8}  {state}")
                if h.service is None:
                    print(f"           {h.cmdline[:78]}")
            if info["unattributed_mb"]:
                print(f"{'(driver)':<10} {'-':>7} {info['unattributed_mb']:>7} {'-':>8}  "
                      "context and overhead")
        if args.need:
            adm = vram.admit(args.need)
            print(f"\nfor a step needing {args.need} MB: {'OK' if adm.ok else 'REFUSED'}")
            print(f"  {adm.reason}")
            if adm.evict:
                print(f"  would free, in order: {' -> '.join(adm.evict)}")
    if args.free:
        after = vram.reclaim()
        print(f"reclaimed — free now: {after if after is not None else 'unknown'} MB")
    return 0


def _cmd_job(args) -> int:
    store = _open_store()
    try:
        job = store.load_job(args.job_id)
        if job is None:
            raise ValueError(f"no job with id {args.job_id}")
        _print_job(job, store)
        print("decision log:")
        for d in store.decisions(job.id):
            scene = "job" if d.scene_index is None else f"s{d.scene_index}"
            score = "" if d.score is None else f" score={d.score}"
            reason = f"  {d.reason}" if d.reason else ""
            print(f"  [{d.created_at}] {scene} {d.phase}/{d.action}{score}{reason}")
        _print_provenance(job)
    finally:
        store.close()
    return 0


def registry_kinds() -> tuple[str, ...]:
    from .registry import KINDS

    return KINDS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vorch", description=__doc__.split("\n")[0])
    # Global, not per-subcommand: every command logs, and the level belongs to the run.
    p.add_argument("--log-level", choices=logsetup.LEVELS, default=None,
                   help="stderr + file log level (default: VORCH_LOG_LEVEL, else info)")
    p.add_argument("--log-file", default=None,
                   help="log file (default: ~/vorch/logs/vorch.log; empty string = stderr only)")
    sub = p.add_subparsers(dest="command", required=True)

    kf = sub.add_parser("keyframe", help="generate a keyframe from a prompt")
    kf.add_argument("prompt")
    kf.add_argument("--width", type=int, default=config.KEYFRAME_WIDTH)
    kf.add_argument("--height", type=int, default=config.KEYFRAME_HEIGHT)
    kf.add_argument("--steps", type=int, default=config.KEYFRAME_STEPS)
    kf.add_argument("--cfg", type=float, default=config.KEYFRAME_CFG)
    kf.add_argument("--seed", type=int, default=config.SEED_RANDOM)
    kf.add_argument("--unet", default=None, help="override the diffusion model name")
    kf.add_argument("--negative", default=None)
    kf.add_argument("--aspect", default=None, help="16:9 | 9:16 | 1:1 | ... (overrides --width/--height)")
    kf.add_argument("--workflow", default="keyframe", help="registry workflow name (kind keyframe)")
    kf.set_defaults(func=_cmd_keyframe)

    an = sub.add_parser("animate", help="animate a keyframe into a video")
    an.add_argument("image", help="local path to the start image")
    an.add_argument("prompt", help="animation/motion prompt")
    an.add_argument("--length", type=int, default=config.WAN_LENGTH, help="frames (4n+1)")
    an.add_argument("--fps", type=int, default=config.WAN_FPS)
    an.add_argument("--seed", type=int, default=config.SEED_RANDOM)
    an.add_argument("--width", type=int, default=config.WAN_WIDTH)
    an.add_argument("--height", type=int, default=config.WAN_HEIGHT)
    an.add_argument("--negative", default=None)
    an.add_argument("--aspect", default=None, help="16:9 | 9:16 | 1:1 | ... (overrides --width/--height)")
    an.add_argument("--workflow", default="animate", help="registry workflow name (kind animate)")
    an.set_defaults(func=_cmd_animate)

    ed = sub.add_parser("edit", help="reference-locked keyframe (identity_edit / kv_edit)")
    ed.add_argument("prompt")
    ed.add_argument("--ref", action="append", required=True, help="reference image (repeatable)")
    ed.add_argument("--workflow", default="identity_edit", help="registry workflow name (kind edit)")
    ed.add_argument("--aspect", default=None, help="fit references + output to 16:9 | 9:16 | 1:1 ...")
    ed.add_argument("--no-fit", action="store_true", help="stage references as-is")
    ed.add_argument("--negative", default=None)
    ed.add_argument("--seed", type=int, default=config.SEED_RANDOM)
    ed.set_defaults(func=_cmd_edit)

    fs = sub.add_parser("faceswap", help="swap a source face onto an image")
    fs.add_argument("input_image", help="local path to the scene/frame")
    fs.add_argument("source_image", help="local path to the master face")
    fs.add_argument("--restore-model", default=config.REACTOR_RESTORE_MODEL)
    fs.add_argument("--workflow", default="faceswap", help="registry workflow name (kind faceswap)")
    fs.set_defaults(func=_cmd_faceswap)

    ev = sub.add_parser("evaluate", help="score an image against a task (Gemma vision)")
    ev.add_argument("image", help="local path to the image to evaluate")
    ev.add_argument("task", help="the intent to judge the image against")
    ev.set_defaults(func=_cmd_evaluate)

    rs = sub.add_parser("run-scene", help="full chain: keyframe -> swap -> animate")
    rs.add_argument("--keyframe-prompt", required=True)
    rs.add_argument("--animation-prompt", required=True)
    rs.add_argument("--master", default=None, help="master face image path (omit with --no-face-swap)")
    rs.add_argument("--no-face-swap", action="store_true", help="animate the keyframe without a face swap")
    rs.add_argument("--seed", type=int, default=config.SEED_RANDOM)
    rs.add_argument("--evaluate", action="store_true", help="judge + retry the keyframe")
    rs.add_argument("--task", default=None, help="eval intent (defaults to keyframe prompt)")
    rs.add_argument("--max-attempts", type=int, default=2)
    rs.add_argument("--aspect", default=None, help="16:9 | 9:16 | 1:1 | ... (default 16:9)")
    rs.add_argument("--length", type=int, default=None, help="Wan frames (4n+1)")
    rs.add_argument("--fps", type=int, default=None)
    rs.set_defaults(func=_cmd_run_scene)

    rq = sub.add_parser("run-sequence", help="chain scenes with last-frame continuity")
    rq.add_argument("scenes", help="YAML file: list of {keyframe_prompt, animation_prompt, continuous?}")
    rq.add_argument("--master", default=None, help="master face image path (omit with --no-face-swap)")
    rq.add_argument("--no-face-swap", action="store_true", help="animate keyframes/frames without a face swap")
    rq.add_argument("--seed", type=int, default=config.SEED_RANDOM)
    rq.add_argument("--evaluate", action="store_true", help="judge + retry fresh keyframes")
    rq.add_argument("--no-relock", action="store_true", help="don't re-swap master onto carried frames")
    rq.add_argument("--max-attempts", type=int, default=2)
    rq.add_argument("--aspect", default=None, help="16:9 | 9:16 | 1:1 | ... (default 16:9)")
    rq.add_argument("--length", type=int, default=None, help="Wan frames (4n+1)")
    rq.add_argument("--fps", type=int, default=None)
    rq.set_defaults(func=_cmd_run_sequence)

    rn = sub.add_parser("run", help="drive a full job from a YAML spec (storyboard -> clips)")
    rn.add_argument("spec", help="job spec YAML (idea, master_face, n_scenes, style, autonomy)")
    rn.set_defaults(func=_cmd_run)

    ap = sub.add_parser("approve", help="clear a semi-auto gate and continue a job")
    ap.add_argument("job_id")
    ap.add_argument("gate", help=f"one of {sorted(GATES)}")
    ap.set_defaults(func=_cmd_approve)

    js = sub.add_parser("jobs", help="list persisted jobs")
    js.set_defaults(func=_cmd_jobs)

    jb = sub.add_parser("job", help="show a job's scenes and decision log")
    jb.add_argument("job_id")
    jb.set_defaults(func=_cmd_job)

    vr = sub.add_parser("vram", help="show GPU memory; --free to reclaim ComfyUI VRAM")
    vr.add_argument("--free", action="store_true", help="POST /free and report new headroom")
    vr.add_argument("--by-process", action="store_true",
                    help="who holds the card: pid, MB, uptime, and whether it can be freed")
    vr.add_argument("--need", type=int, default=None, metavar="MB",
                    help="with --by-process: would a step needing this much be admitted?")
    vr.set_defaults(func=_cmd_vram)

    sv = sub.add_parser("serve", help="run the HTTP API + SSE server")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.set_defaults(func=_cmd_serve)

    oa = sub.add_parser("openapi", help="print the API's OpenAPI schema (JSON) for the typed client")
    oa.set_defaults(func=_cmd_openapi)

    st = sub.add_parser("setup", help="environment check: database, ComfyUI, Gemma, tools, GPU, workflows")
    sts = st.add_subparsers(dest="setup_command", required=True)
    stc = sts.add_parser("check", help="run every check (exit 1 on any FAIL)")
    stc.add_argument("--deep", action="store_true", help="also probe Gemma vision (--mmproj)")
    stc.add_argument("--profile", default=None, help="check this profile's workflows strictly")
    st.set_defaults(func=_cmd_setup)

    pf = sub.add_parser("profiles", help="list output profiles (built-in + user); --import adds one from YAML")
    pf.add_argument("--import", dest="import_file", default=None, metavar="FILE",
                    help="create a user profile from a YAML {name, base, overrides} (e.g. modal/profiles/*.yaml)")
    pf.set_defaults(func=_cmd_profiles)

    pj = sub.add_parser("projects", help="list projects")
    pj.set_defaults(func=_cmd_projects)

    pr = sub.add_parser(
        "project",
        help="project: add <name> | add-asset <slug> <path> --kind | label-assets <slug> | set-kind <slug> <asset> <kind> | show <slug>",
    )
    prs = pr.add_subparsers(dest="project_command", required=True)
    pa = prs.add_parser("add")
    pa.add_argument("name")
    pa.add_argument("--profile", default="cinematic_16x9", help="default profile id")
    pa.add_argument("--export-root", default=None)
    pas = prs.add_parser("add-asset")
    pas.add_argument("slug")
    pas.add_argument("path")
    pas.add_argument("--kind", default="master_face", choices=["master_face", "reference", "product", "logo", "other", "video"])
    pas.add_argument("--label", default=None)
    pla = prs.add_parser("label-assets", help="describe assets with Gemma and propose labels for review")
    pla.add_argument("slug")
    pla.add_argument("--kind", default=None, choices=["master_face", "reference", "product", "logo", "other", "video"])
    pla.add_argument("--hint", default=None,
                     help="what this collection is, and the tag vocabulary to use (steers `tags`); "
                          "defaults to the project's describe hint")
    pla.add_argument("--force", action="store_true", help="redo assets that already have a suggested label")
    pla.add_argument("--overwrite-human", action="store_true", help="also replace labels a person wrote")
    pla.add_argument("--relabel", action="store_true", help="rebuild labels from stored notes; no model calls")
    pla.add_argument("--dry-run", action="store_true", help="print proposals, write nothing")
    pla.add_argument("--limit", type=int, default=None)
    psk = prs.add_parser("set-kind", help="change what an asset is used as; a suggested label is rebuilt")
    psk.add_argument("slug")
    psk.add_argument("asset_id")
    psk.add_argument("kind", choices=["master_face", "reference", "product", "logo", "other", "video"])
    psh = prs.add_parser("show")
    psh.add_argument("slug")
    pr.set_defaults(func=_cmd_project)

    asm = sub.add_parser("assemble", help="join a job's clips into one video with xfade transitions")
    asm.add_argument("job_id")
    asm.add_argument("--transition", default="fade", help="transition for every cut (or 'none'); see `ffmpeg -h filter=xfade`")
    asm.add_argument("--duration", type=float, default=0.5, help="transition seconds")
    asm.add_argument("--spec", default=None, help="JSON file with {items, cuts, fps, ...} (overrides --transition/--duration)")
    asm.add_argument("--fps", type=int, default=None)
    asm.add_argument("--encoder", default=None, choices=["libx264", "h264_nvenc"])
    asm.add_argument("--delivery", default=None,
                     help="master to a delivery target: source | master_1080 | social_1080 | youtube_1080")
    asm.add_argument("--width", type=int, default=None, help="explicit delivery width (overrides --delivery)")
    asm.add_argument("--height", type=int, default=None, help="explicit delivery height")
    asm.add_argument("--grain", type=float, default=None, help="0-3: light film grain, hides upscaling")
    asm.add_argument("--force", action="store_true", help="assemble even if the job is not done")
    asm.add_argument("--list", action="store_true", help="list the job's assembly versions (* = current)")
    asm.add_argument("--select", type=int, default=None, metavar="N", help="make version N the current assembly")
    asm.set_defaults(func=_cmd_assemble)

    ex = sub.add_parser("export", help="write the Shotcut handoff folder for a finished job")
    ex.add_argument("job_id")
    ex.add_argument("--force", action="store_true", help="export even if the job is not done")
    ex.set_defaults(func=_cmd_export)

    rg = sub.add_parser("registry", help="workflow registry: list | validate | inspect | add")
    rgs = rg.add_subparsers(dest="registry_command", required=True)
    rl = rgs.add_parser("list", help="registered workflows (built-in + user)")
    rl.add_argument("--kind", default=None)
    rv = rgs.add_parser("validate", help="check YAML vs template; --online also checks ComfyUI")
    rv.add_argument("name", nargs="?", default=None)
    rv.add_argument("--online", action="store_true")
    rv.add_argument("--backend", default=None, help="check against this backend's ComfyUI instead of each workflow's own")
    ri = rgs.add_parser("inspect", help="list a template's nodes + literal inputs with role guesses")
    ri.add_argument("file")
    ri.add_argument("--online", action="store_true", help="show installed options per loader")
    ri.add_argument("--backend", default=None, help="backend whose ComfyUI to ask (default local)")
    ra = rgs.add_parser("add", help="scaffold a registry YAML for a new template into the user dir")
    ra.add_argument("file")
    ra.add_argument("--name", default=None)
    ra.add_argument("--kind", default=None, choices=list(registry_kinds()))
    ra.add_argument("--out", default=None, help="directory (default: VORCH_USER_WORKFLOW_DIR)")
    ra.add_argument("--force", action="store_true")
    rg.set_defaults(func=_cmd_registry)

    upp = sub.add_parser("up", help="start the services, in dependency order (adopts what is running)")
    upp.add_argument("names", nargs="*", help="only these (dependencies are pulled in)")
    upp.add_argument("--no-wait", action="store_true", help="do not wait for health probes")
    upp.add_argument("--dry-run", action="store_true", help="print what would happen")
    upp.add_argument("--reclaim-orphans", action="store_true",
                     help="stop services left behind by a vorch that was killed")
    upp.set_defaults(func=_cmd_up)

    dwn = sub.add_parser("down", help="stop the services vorch started")
    dwn.add_argument("names", nargs="*")
    dwn.add_argument("--phase-only", action="store_true", help="only the on-demand ones (e.g. tts)")
    dwn.add_argument("--force", action="store_true",
                     help="also stop services started outside vorch")
    dwn.set_defaults(func=_cmd_down)

    st = sub.add_parser("status", help="which services are up, which are ours, and who holds the GPU")
    st.add_argument("--json", action="store_true")
    st.add_argument("--no-probe", action="store_true", help="skip the health requests (faster)")
    st.set_defaults(func=_cmd_status)

    lg = sub.add_parser("logs", help="tail a service's log file")
    lg.add_argument("name")
    lg.add_argument("-n", "--lines", type=int, default=200)
    lg.set_defaults(func=_cmd_logs)

    sv2 = sub.add_parser("services", help="the service manifest: which processes vorch depends on")
    sv2s = sv2.add_subparsers(dest="services_command", required=True)
    sl = sv2s.add_parser("list", help="every service, its measured VRAM cost and its start command")
    sl.add_argument("--json", action="store_true", help="machine-readable, with resolved argv/env")
    sv2.set_defaults(func=_cmd_services)

    dbp = sub.add_parser("db", help="database: init | status | import-sqlite <file>")
    dbs = dbp.add_subparsers(dest="db_command", required=True)
    dbs.add_parser("init", help="create the database if missing and apply migrations")
    dbs.add_parser("status", help="show DSN, schema version and job count")
    imp = dbs.add_parser("import-sqlite", help="copy jobs + decisions from a legacy runs/*.db")
    imp.add_argument("path", help="path to the SQLite file (e.g. runs/vorch.db)")
    dbp.set_defaults(func=_cmd_db)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logsetup.configure(args.log_level, log_file=args.log_file)
    try:
        return args.func(args)
    except (
        comfy.ComfyError, EvaluationError, MediaError, GemmaError, StoryboardError,
        FileNotFoundError, ValueError, DatabaseUnavailable,
    ) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
