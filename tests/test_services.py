"""The service manifest (vorch/services.py) — pure parsing, no process ever started.

These tests are the reason the module is split from the supervisor: the description of the world
is checkable without a GPU, a network or a subprocess.
"""

from __future__ import annotations

import pytest
import yaml

from vorch import config, services


@pytest.fixture(autouse=True)
def _clean_cache():
    services.reset_cache()
    yield
    services.reset_cache()


def write(tmp_path, doc: dict):
    p = tmp_path / "services.yaml"
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return p


# --- the shipped manifest ------------------------------------------------------------


def test_builtin_manifest_parses_and_orders():
    # An absent user file is not an error: the built-ins stand on their own.
    specs = services.load("/nonexistent/services.yaml", force=True)
    assert set(specs) == {"postgres", "gemma", "comfy", "tts", "api", "ui"}
    order = services.order(specs)
    # Dependencies come first: the API opens a pool, the UI proxies to the API.
    assert order.index("postgres") < order.index("api") < order.index("ui")
    assert all(s.builtin for s in specs.values())


def test_builtin_vram_figures_are_the_measured_ones():
    """These numbers drive every eviction decision; a silent edit must fail a test."""
    specs = services.load("/nonexistent/services.yaml", force=True)
    assert specs["gemma"].vram.holds_mb == 1838, "Gemma costs 1838 MiB even with --n-gpu-layers 0"
    assert specs["tts"].vram.holds_mb == 11268, "OmniVoice is 11268 MiB, not the old ~5.3 GB claim"
    assert specs["comfy"].vram.needs_mb == 22000
    # The arithmetic that makes narration and a render mutually exclusive on a 16303 MiB card.
    assert specs["tts"].vram.holds_mb + specs["comfy"].vram.needs_mb > 16303
    assert specs["tts"].vram.exclusive and specs["comfy"].vram.elastic
    assert specs["gemma"].vram.kind == "resident"


def test_launchers_that_exec_or_fork_pin_their_matcher():
    """A derived matcher cannot find a process whose argv does not survive the launch."""
    specs = services.load("/nonexistent/services.yaml", force=True)
    # serve_bg.sh execs `python -m omnivoice.cli.serve`; `comfy launch` forks `python main.py`,
    # and that grandchild is what holds the VRAM; `npm run dev` becomes `next-server`.
    assert specs["tts"].matcher().cmdline_contains == ("omnivoice.cli.serve",)
    assert specs["comfy"].matcher().cmdline_contains == ("main.py",)
    assert specs["ui"].matcher().cmdline_contains == ("next-server",)
    # gemma's argv ends in the port number, so a naive "last token" needle would be "8080".
    assert specs["gemma"].matcher().cmdline_contains == ("llama-server",)


def test_tts_carries_the_voice_and_can_report_it():
    spec = services.load("/nonexistent/services.yaml", force=True)["tts"]
    assert spec.env["OMNIVOICE_BG_REF_AUDIO"] == "${VORCH_TTS_REF_AUDIO}"
    assert spec.adopt.verify == "env", "the wrong-voice bug is only detectable by reading /proc/environ"
    assert spec.adopt.on_unverified == "warn", "a hand-started server must not be blocked"
    resolved = spec.env_resolved()
    assert resolved["OMNIVOICE_BG_REF_AUDIO"] == config.TTS_REF_AUDIO
    assert len(resolved["OMNIVOICE_BG_REF_TEXT"]) > 20  # @file: inlined the transcript
    # A long transcript is digested so a pidfile or log header stays readable.
    assert spec.identity()["OMNIVOICE_BG_REF_TEXT"].startswith("sha1:")


# --- value resolution ----------------------------------------------------------------


def test_vorch_vars_fall_back_to_config_when_unset(monkeypatch):
    monkeypatch.delenv("VORCH_COMFY_URL", raising=False)
    assert services.resolve_vars("${VORCH_COMFY_URL}/x", where="t") == f"{config.COMFY_URL}/x"
    monkeypatch.setenv("VORCH_COMFY_URL", "http://elsewhere:9999")
    assert services.resolve_vars("${VORCH_COMFY_URL}", where="t") == "http://elsewhere:9999"


def test_unresolvable_var_is_an_error():
    with pytest.raises(services.ServiceError, match=r"NOPE_NOT_SET"):
        services.resolve_vars("${NOPE_NOT_SET}", where="somewhere")


def test_at_file_is_read_late_and_relative_to_cwd(tmp_path):
    (tmp_path / "voice.txt").write_text("  a transcript, verbatim  \n", encoding="utf-8")
    spec = services.parse_service("t", {
        "manage": "full", "argv": ["true"], "cwd": str(tmp_path),
        "env": {"REF": "@file:voice.txt"},
    })
    assert spec.env_resolved()["REF"] == "a transcript, verbatim"


def test_missing_at_file_raises_only_when_used(tmp_path):
    """Parsing must not fail on a file that is not there yet — starting must."""
    spec = services.parse_service("t", {
        "manage": "full", "argv": ["true"], "cwd": str(tmp_path),
        "env": {"REF": "@file:absent.txt"},
    })
    with pytest.raises(services.ServiceError, match="cannot read"):
        spec.env_resolved()
    # to_dict reports it instead of raising, because it feeds the CLI and the API.
    assert "env_error" in spec.to_dict(resolve=True)["resolved"]


# --- user overrides ------------------------------------------------------------------


def test_user_file_overrides_one_field_without_restating_the_entry(tmp_path):
    p = write(tmp_path, {"services": {"tts": {"idle_timeout": 30}}})
    specs = services.load(p, force=True)
    tts = specs["tts"]
    assert tts.idle_timeout == 30
    assert tts.builtin is False and tts.source == str(p)
    # Everything else survives the merge, including nested mappings.
    assert tts.vram.holds_mb == 11268
    assert tts.argv == ("bash", "serve_bg.sh")
    assert tts.adopt.verify == "env"


def test_user_file_merges_nested_mappings(tmp_path):
    p = write(tmp_path, {"services": {"tts": {"vram": {"needs_mb": 12000},
                                             "adopt": {"on_unverified": "refuse"}}}})
    tts = services.load(p, force=True)["tts"]
    assert tts.vram.needs_mb == 12000
    assert tts.vram.holds_mb == 11268, "the untouched key must survive"
    assert tts.adopt.on_unverified == "refuse"
    assert tts.adopt.verify == "env"


def test_user_file_can_demote_a_service_to_monitor_only(tmp_path):
    """Setting every service to adopt is a legitimate configuration: today's behaviour + diagnostics."""
    p = write(tmp_path, {"services": {n: {"manage": "adopt"} for n in
                                      ("gemma", "comfy", "tts", "api")}})
    specs = services.load(p, force=True)
    assert not any(s.managed for s in specs.values())


def test_user_file_can_add_a_service(tmp_path):
    p = write(tmp_path, {"services": {"extra": {
        "manage": "full", "argv": ["sleep", "1"],
        "health": {"probe": "http", "url": "http://localhost:1/x"}}}})
    specs = services.load(p, force=True)
    assert "extra" in specs and specs["extra"].managed


def test_cache_is_invalidated_when_the_user_file_changes(tmp_path):
    p = write(tmp_path, {"services": {"tts": {"idle_timeout": 30}}})
    assert services.load(p)["tts"].idle_timeout == 30
    write(tmp_path, {"services": {"tts": {"idle_timeout": 99}}})
    assert services.load(p)["tts"].idle_timeout == 99


# --- validation ----------------------------------------------------------------------


def test_bad_name_rejected():
    with pytest.raises(services.ServiceError, match="invalid"):
        services.parse_service("Not A Name", {"manage": "adopt"})


@pytest.mark.parametrize("field,value,message", [
    ("manage", "sometimes", "manage must be one of"),
    ("lifetime", "forever", "lifetime must be one of"),
    ("ready_timeout", "soon", "must be a number"),
    ("ready_timeout", -1, "must be >= 0"),
    ("depends_on", "gemma", "must be a list"),
    ("env", ["a"], "must be a mapping"),
])
def test_field_validation(field, value, message):
    with pytest.raises(services.ServiceError, match=message):
        services.parse_service("t", {"manage": "adopt", "argv": ["true"], field: value})


def test_manage_full_needs_an_argv():
    with pytest.raises(services.ServiceError, match="needs an argv"):
        services.parse_service("t", {"manage": "full"})


def test_warm_needs_an_idle_timeout():
    """Otherwise 'warm' is just 'left running' — which is the bug, not the feature."""
    with pytest.raises(services.ServiceError, match="needs an idle_timeout"):
        services.parse_service("t", {"manage": "full", "argv": ["true"], "lifetime": "warm"})


def test_elastic_needs_a_free_endpoint():
    with pytest.raises(services.ServiceError, match="needs a free_endpoint"):
        services.parse_service("t", {"manage": "adopt", "vram": {"class": "elastic"}})


def test_vram_class_none_cannot_want_memory():
    with pytest.raises(services.ServiceError, match="cannot also want memory"):
        services.parse_service("t", {"manage": "adopt", "vram": {"class": "none", "holds_mb": 10}})


def test_http_probe_needs_a_url():
    with pytest.raises(services.ServiceError, match="needs a url"):
        services.parse_service("t", {"manage": "adopt", "health": {"probe": "http"}})


# --- dependency ordering --------------------------------------------------------------


def test_unknown_dependency_is_an_error(tmp_path):
    p = write(tmp_path, {"services": {"api": {"depends_on": ["ghost"]}}})
    with pytest.raises(services.ServiceError, match="depends on 'ghost'"):
        services.load(p, force=True)


def test_dependency_cycle_is_named(tmp_path):
    p = write(tmp_path, {"services": {"gemma": {"depends_on": ["comfy"]},
                                      "comfy": {"depends_on": ["gemma"]}}})
    with pytest.raises(services.ServiceError, match="dependency cycle"):
        services.load(p, force=True)


def test_self_dependency_is_an_error(tmp_path):
    p = write(tmp_path, {"services": {"api": {"depends_on": ["api"]}}})
    with pytest.raises(services.ServiceError, match="depends on itself"):
        services.load(p, force=True)


def test_order_of_a_subset_still_pulls_in_dependencies():
    specs = services.load("/nonexistent/services.yaml", force=True)
    assert services.order(specs, ["ui"]) == ["postgres", "api", "ui"]


def test_order_rejects_an_unknown_name():
    specs = services.load("/nonexistent/services.yaml", force=True)
    with pytest.raises(services.ServiceError, match="unknown service"):
        services.order(specs, ["ghost"])


# --- the shape the CLI and API consume -----------------------------------------------


def test_to_dict_resolved_gives_a_command_you_could_run_yourself():
    """Hand-starting is first class, so the manifest has to hand over the command."""
    spec = services.load("/nonexistent/services.yaml", force=True)["gemma"]
    d = spec.to_dict(resolve=True)
    argv = d["resolved"]["argv"]
    assert argv[0].endswith("llama-server")
    assert "--mmproj" in argv, "without the projector every verdict is meaningless"
    assert "~" not in " ".join(argv) and d["resolved"]["cwd"].startswith("/")
    assert d["resolved"]["health_url"].endswith("/v1/models")
    assert d["vram"]["class"] == "resident"


def test_get_reports_unknown_names_with_the_file_to_edit():
    with pytest.raises(services.ServiceError, match="unknown service 'ghost'"):
        services.get("ghost")
