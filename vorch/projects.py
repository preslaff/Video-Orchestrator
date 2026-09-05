"""Projects / campaigns — shared assets, default profile, export root.

A project groups jobs and owns the files they share: the master face, identity
references, product shots, logos. Assets are copied under
``config.PROJECTS_ROOT/<slug>/assets/`` (content-hashed) and referenced by id.
The Project itself is one JSON blob in the ``projects`` table; the ``slug`` is
unique and doubles as the export folder name.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import unicodedata
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from . import config

AssetKind = Literal["master_face", "reference", "product", "logo", "other"]
ASSET_KINDS = ("master_face", "reference", "product", "logo", "other")

_CYRILLIC = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ж": "zh", "з": "z", "и": "i",
    "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s",
    "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sht",
    "ъ": "a", "ь": "", "ю": "yu", "я": "ya",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(text: str, *, max_len: int = 40) -> str:
    """ASCII slug: Cyrillic transliterated, accents folded, ``[a-z0-9-]`` only."""
    text = "".join(_CYRILLIC.get(ch, ch) for ch in text.lower())
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    text = re.sub(r"-{2,}", "-", text)
    return text[:max_len].rstrip("-") or "untitled"


@dataclass
class Asset:
    id: str
    kind: str
    name: str            # original file name
    path: str            # absolute path under the project dir
    label: str = ""
    sha256: str = ""
    created_at: str = ""


@dataclass
class Project:
    id: str
    slug: str
    name: str
    brand_notes: str = ""
    style_notes: str = ""
    default_profile_id: str = "cinematic_16x9"
    export_root: str = ""             # "" -> config.EXPORT_ROOT
    assets: list[Asset] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

    def asset(self, asset_id: str) -> Asset | None:
        return next((a for a in self.assets if a.id == asset_id), None)

    def assets_of(self, kind: str) -> list[Asset]:
        return [a for a in self.assets if a.kind == kind]


def new_project(name: str, **kw: Any) -> Project:
    ts = _now()
    return Project(id=uuid.uuid4().hex, slug=kw.pop("slug", None) or slugify(name), name=name,
                   created_at=ts, updated_at=ts, **kw)


def project_dir(project: Project) -> Path:
    return config.PROJECTS_ROOT / project.slug


def export_root(project: Project | None) -> Path:
    if project is not None and project.export_root:
        return Path(project.export_root).expanduser()
    return config.EXPORT_ROOT


def master_face(project: Project | None) -> Asset | None:
    """The project's default identity (first master_face asset)."""
    if project is None:
        return None
    faces = project.assets_of("master_face")
    return faces[0] if faces else None


def identity_assets(project: Project | None) -> list[Asset]:
    """All faces a scene may be assigned (label them by who/when: "Maria, age 8")."""
    return project.assets_of("master_face") if project is not None else []


def resolve_asset_paths(project: Project | None, asset_ids: Sequence[str]) -> list[str]:
    """Asset ids -> file paths, in order. KeyError names the first unknown id."""
    if not asset_ids:
        return []
    if project is None:
        raise KeyError("no project")
    out: list[str] = []
    for aid in asset_ids:
        a = project.asset(aid)
        if a is None:
            raise KeyError(aid)
        out.append(a.path)
    return out


def set_asset_label(project: Project, asset_id: str, label: str) -> Asset:
    a = project.asset(asset_id)
    if a is None:
        raise KeyError(asset_id)
    a.label = label.strip()
    project.updated_at = _now()
    return a


def add_asset(
    project: Project,
    source: str | Path | bytes,
    *,
    kind: str,
    name: str,
    label: str = "",
) -> Asset:
    """Copy a file (path or bytes) into the project's asset dir and register it."""
    if kind not in ASSET_KINDS:
        raise ValueError(f"asset kind must be one of {ASSET_KINDS}, got {kind!r}")
    safe = slugify(Path(name).stem, max_len=40) or "asset"
    suffix = Path(name).suffix.lower()
    aid = uuid.uuid4().hex[:12]
    dest_dir = project_dir(project) / "assets"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{aid}_{safe}{suffix}"
    if isinstance(source, bytes):
        dest.write_bytes(source)
    else:
        src = Path(source).expanduser()
        if not src.is_file():
            raise FileNotFoundError(f"asset source not found: {src}")
        shutil.copy2(src, dest)
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    asset = Asset(id=aid, kind=kind, name=Path(name).name, path=str(dest), label=label,
                  sha256=digest, created_at=_now())
    project.assets.append(asset)
    project.updated_at = _now()
    return asset


def remove_asset(project: Project, asset_id: str, *, delete_file: bool = True) -> bool:
    a = project.asset(asset_id)
    if a is None:
        return False
    project.assets.remove(a)
    if delete_file:
        try:
            Path(a.path).unlink()
        except OSError:
            pass
    project.updated_at = _now()
    return True


# --- (de)serialization ----------------------------------------------------------------


def asset_to_dict(a: Asset) -> dict:
    return {"id": a.id, "kind": a.kind, "name": a.name, "path": a.path, "label": a.label,
            "sha256": a.sha256, "created_at": a.created_at}


def asset_from_dict(d: dict) -> Asset:
    return Asset(id=d["id"], kind=d.get("kind", "other"), name=d.get("name", ""), path=d["path"],
                 label=d.get("label", ""), sha256=d.get("sha256", ""), created_at=d.get("created_at", ""))


def project_to_dict(p: Project) -> dict:
    return {
        "id": p.id, "slug": p.slug, "name": p.name, "brand_notes": p.brand_notes,
        "style_notes": p.style_notes, "default_profile_id": p.default_profile_id,
        "export_root": p.export_root, "assets": [asset_to_dict(a) for a in p.assets],
        "created_at": p.created_at, "updated_at": p.updated_at,
    }


def project_from_dict(d: dict) -> Project:
    return Project(
        id=d["id"], slug=d["slug"], name=d.get("name", ""), brand_notes=d.get("brand_notes", ""),
        style_notes=d.get("style_notes", ""), default_profile_id=d.get("default_profile_id", "cinematic_16x9"),
        export_root=d.get("export_root", ""), assets=[asset_from_dict(a) for a in d.get("assets", []) or []],
        created_at=d.get("created_at", ""), updated_at=d.get("updated_at", ""),
    )


# --- store access ----------------------------------------------------------------------


class SlugTaken(ValueError):
    pass


def save(store: Any, project: Project) -> Project:
    project.updated_at = _now()
    if not project.created_at:
        project.created_at = project.updated_at
    store.save_project(project_to_dict(project))
    return project


def load(store: Any, project_id: str) -> Project | None:
    d = store.load_project(project_id)
    return project_from_dict(d) if d else None


def load_by_slug(store: Any, slug: str) -> Project | None:
    d = store.load_project_by_slug(slug)
    return project_from_dict(d) if d else None


def list_all(store: Any) -> list[Project]:
    return [project_from_dict(d) for d in store.list_projects()]
