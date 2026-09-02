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
from pathlib import Path

from . import config, formats, provenance
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
        profile=profiles.from_legacy_spec(spec),
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

    from .server import create_app

    app = create_app(_open_store())
    print(f"API on http://{args.host}:{args.port}/api  (docs: /docs)")
    print(f"UI: {config.UI_URL}  (run `npm run dev` in frontend/)")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def _cmd_openapi(args) -> int:
    from .api import openapi_schema

    json.dump(openapi_schema(), sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


def _cmd_dashboard(args) -> int:
    from .app import run  # lazy: NiceGUI only needed for the dashboard

    print(f"dashboard on http://{args.host}:{args.port}")
    run(host=args.host, port=args.port)
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
    from . import profiles

    store = _open_store()
    try:
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
        if args.project_command == "show":
            print(f"{p.name} [{p.slug}]  profile={p.default_profile_id}  export={projects.export_root(p)}")
            for a in p.assets:
                print(f"  {a.kind:<12} {a.id}  {a.name}  {a.label}")
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
        info = comfy.object_info() if args.online else None
        specs = [reg.get(args.name)] if args.name else reg.list()
        bad = 0
        for spec in specs:
            problems = reg.validate(spec, object_info=info)
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
        info = comfy.object_info() if args.online else None
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

    pf = sub.add_parser("profiles", help="list output profiles (built-in + user)")
    pf.set_defaults(func=_cmd_profiles)

    pj = sub.add_parser("projects", help="list projects")
    pj.set_defaults(func=_cmd_projects)

    pr = sub.add_parser("project", help="project: add <name> | add-asset <slug> <path> --kind | show <slug>")
    prs = pr.add_subparsers(dest="project_command", required=True)
    pa = prs.add_parser("add")
    pa.add_argument("name")
    pa.add_argument("--profile", default="cinematic_16x9", help="default profile id")
    pa.add_argument("--export-root", default=None)
    pas = prs.add_parser("add-asset")
    pas.add_argument("slug")
    pas.add_argument("path")
    pas.add_argument("--kind", default="master_face", choices=["master_face", "reference", "product", "logo", "other"])
    pas.add_argument("--label", default=None)
    psh = prs.add_parser("show")
    psh.add_argument("slug")
    pr.set_defaults(func=_cmd_project)

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
    ri = rgs.add_parser("inspect", help="list a template's nodes + literal inputs with role guesses")
    ri.add_argument("file")
    ri.add_argument("--online", action="store_true", help="show installed options per loader")
    ra = rgs.add_parser("add", help="scaffold a registry YAML for a new template into the user dir")
    ra.add_argument("file")
    ra.add_argument("--name", default=None)
    ra.add_argument("--kind", default=None, choices=list(registry_kinds()))
    ra.add_argument("--out", default=None, help="directory (default: VORCH_USER_WORKFLOW_DIR)")
    ra.add_argument("--force", action="store_true")
    rg.set_defaults(func=_cmd_registry)

    dbp = sub.add_parser("db", help="database: init | status | import-sqlite <file>")
    dbs = dbp.add_subparsers(dest="db_command", required=True)
    dbs.add_parser("init", help="create the database if missing and apply migrations")
    dbs.add_parser("status", help="show DSN, schema version and job count")
    imp = dbs.add_parser("import-sqlite", help="copy jobs + decisions from a legacy runs/*.db")
    imp.add_argument("path", help="path to the SQLite file (e.g. runs/vorch.db)")
    dbp.set_defaults(func=_cmd_db)

    db = sub.add_parser("dashboard", help="run the NiceGUI web dashboard")
    db.add_argument("--host", default="127.0.0.1")
    db.add_argument("--port", type=int, default=8090)
    db.set_defaults(func=_cmd_dashboard)

    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
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
