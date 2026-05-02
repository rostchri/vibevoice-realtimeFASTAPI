"""Factory – instantiates the correct adapter for a given model profile."""

from __future__ import annotations

from typing import Any

from runner.adapters.base import EngineAdapter
from runner.adapters.longform_native import LongformNativeAdapter
from runner.adapters.realtime_demo import RealtimeDemoAdapter
from runner.errors import UnknownModelError
from runner.model_registry import ModelProfile, get_model_profile


def make_adapter(model_key: str, **kwargs: Any) -> EngineAdapter:
    """Create an :class:`EngineAdapter` for the given canonical *model_key*.

    Extra *kwargs* are forwarded to the adapter constructor (e.g.
    ``model_path``, ``device``).
    """
    profile: ModelProfile = get_model_profile(model_key)

    if profile.family == "realtime" and profile.loader_mode == "subprocess_demo":
        return RealtimeDemoAdapter(profile, **kwargs)

    if profile.family == "longform" and profile.loader_mode == "native_longform":
        return LongformNativeAdapter(profile, **kwargs)

    raise UnknownModelError(
        f"No adapter registered for family={profile.family!r}, "
        f"loader_mode={profile.loader_mode!r}"
    )
