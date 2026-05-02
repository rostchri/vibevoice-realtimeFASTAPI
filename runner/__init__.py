"""Runner package – multi-model TTS backend registry and adapter layer."""

from runner.adapter_factory import make_adapter
from runner.errors import (
    BackendUnavailableError,
    CapabilityError,
    InvalidRequestForModelError,
    UnknownModelError,
)
from runner.model_registry import ModelProfile, get_model_profile, resolve_model_key
from runner.types import SpeakerTurn, SpeechRequest

__all__ = [
    "BackendUnavailableError",
    "CapabilityError",
    "InvalidRequestForModelError",
    "ModelProfile",
    "SpeakerTurn",
    "SpeechRequest",
    "UnknownModelError",
    "get_model_profile",
    "make_adapter",
    "resolve_model_key",
]
