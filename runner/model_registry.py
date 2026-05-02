"""Model registry – canonical profiles, aliases, and lookup helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from runner.errors import UnknownModelError


@dataclass(frozen=True)
class ModelProfile:
    """Describes a supported VibeVoice model variant."""

    key: str
    hf_model_id: str
    default_local_dir: str
    family: Literal["realtime", "longform"]
    loader_mode: Literal["subprocess_demo", "native_longform"]
    supports_stream: bool
    supports_multispeaker: bool
    supports_voice_list: bool


# ---------------------------------------------------------------------------
# Canonical model profiles
# ---------------------------------------------------------------------------

_PROFILES: dict[str, ModelProfile] = {
    "realtime-0.5b": ModelProfile(
        key="realtime-0.5b",
        hf_model_id="microsoft/VibeVoice-Realtime-0.5B",
        default_local_dir="models/VibeVoice-Realtime-0.5B",
        family="realtime",
        loader_mode="subprocess_demo",
        supports_stream=True,
        supports_multispeaker=False,
        supports_voice_list=True,
    ),
    "tts-1.5b": ModelProfile(
        key="tts-1.5b",
        hf_model_id="microsoft/VibeVoice-1.5B",
        default_local_dir="models/VibeVoice-1.5B",
        family="longform",
        loader_mode="native_longform",
        supports_stream=False,
        supports_multispeaker=True,
        supports_voice_list=False,
    ),
    "tts-7b": ModelProfile(
        key="tts-7b",
        hf_model_id=os.environ.get("VIBEVOICE_7B_MODEL_ID", "microsoft/VibeVoice-7B"),
        default_local_dir="models/VibeVoice-7B",
        family="longform",
        loader_mode="native_longform",
        supports_stream=False,
        supports_multispeaker=True,
        supports_voice_list=False,
    ),
}

# ---------------------------------------------------------------------------
# Alias mapping – maps convenience / OpenAI-compat names to canonical keys
# ---------------------------------------------------------------------------

_ALIASES: dict[str, str] = {
    # OpenAI-compat defaults → realtime for backward compatibility
    "tts-1": "realtime-0.5b",
    "tts-1-hd": "realtime-0.5b",
    # Friendly long names
    "vibevoice-realtime-0.5b": "realtime-0.5b",
    "vibevoice-1.5b": "tts-1.5b",
    "vibevoice-7b": "tts-7b",
}

DEFAULT_MODEL_KEY = "realtime-0.5b"


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def list_model_keys() -> list[str]:
    """Return all canonical model keys."""
    return list(_PROFILES.keys())


def list_aliases() -> dict[str, str]:
    """Return a copy of the alias mapping."""
    return dict(_ALIASES)


def resolve_model_key(requested: str | None) -> str:
    """Resolve *requested* (model name, alias, or ``None``) to a canonical key.

    Raises :class:`UnknownModelError` if the value cannot be mapped.
    """
    if requested is None or requested.strip() == "":
        return DEFAULT_MODEL_KEY

    normalised = requested.strip().lower()

    # Direct canonical hit
    if normalised in _PROFILES:
        return normalised

    # Alias hit
    if normalised in _ALIASES:
        return _ALIASES[normalised]

    raise UnknownModelError(normalised)


def get_model_profile(model_key: str) -> ModelProfile:
    """Return the :class:`ModelProfile` for a canonical *model_key*.

    Raises :class:`UnknownModelError` if the key is not registered.
    """
    try:
        return _PROFILES[model_key]
    except KeyError:
        raise UnknownModelError(model_key) from None
