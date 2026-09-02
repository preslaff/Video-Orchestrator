"""Output profiles: built-ins (read-only) + user profiles (CRUD, clone)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ValidationError

from .. import profiles
from ..profiles import OutputProfile
from ..state import Store
from .deps import get_store
from .errors import ApiError

router = APIRouter(prefix="/api/profiles", tags=["profiles"])


class ProfileCreate(BaseModel):
    """A new user profile: ``base`` (id to start from, default cinematic_16x9) + overrides."""

    name: str
    base: str | None = None
    overrides: dict[str, Any] | None = None


class ProfileClone(BaseModel):
    name: str
    overrides: dict[str, Any] | None = None


def _load(store: Store, profile_id: str) -> OutputProfile:
    try:
        return profiles.resolve(store, profile_id)
    except KeyError:
        raise ApiError(404, "not_found", f"no such profile {profile_id!r}")


def _writable(p: OutputProfile) -> OutputProfile:
    if p.builtin:
        raise ApiError(403, "forbidden", f"profile {p.id!r} is built-in; clone it instead")
    return p


def _validated(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except (ValidationError, ValueError) as e:
        raise ApiError(422, "validation_error", "invalid profile", details=str(e))


@router.get("", response_model=list[OutputProfile])
async def list_profiles(store: Store = Depends(get_store)) -> list[OutputProfile]:
    return profiles.list_all(store)


@router.post("", response_model=OutputProfile, status_code=201)
async def create_profile(body: ProfileCreate, store: Store = Depends(get_store)) -> OutputProfile:
    base = _load(store, body.base or profiles.DEFAULT_PROFILE_ID)
    p = _validated(profiles.clone, base, body.name, overrides=body.overrides)
    return profiles.save(store, p)


@router.get("/{profile_id}", response_model=OutputProfile)
async def get_profile(profile_id: str, store: Store = Depends(get_store)) -> OutputProfile:
    return _load(store, profile_id)


@router.put("/{profile_id}", response_model=OutputProfile)
async def update_profile(profile_id: str, patch: dict[str, Any], store: Store = Depends(get_store)) -> OutputProfile:
    p = _writable(_load(store, profile_id))
    patch = {k: v for k, v in patch.items() if k not in ("id", "builtin", "created_at", "updated_at")}
    updated = _validated(profiles.apply_overrides, p, patch)
    return profiles.save(store, updated)


@router.delete("/{profile_id}", status_code=204)
async def delete_profile(profile_id: str, store: Store = Depends(get_store)) -> None:
    _writable(_load(store, profile_id))
    store.delete_profile(profile_id)


@router.post("/{profile_id}/clone", response_model=OutputProfile, status_code=201)
async def clone_profile(profile_id: str, body: ProfileClone, store: Store = Depends(get_store)) -> OutputProfile:
    base = _load(store, profile_id)
    p = _validated(profiles.clone, base, body.name, overrides=body.overrides)
    return profiles.save(store, p)
