"""Abstract base class for TTS engine adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from runner.model_registry import ModelProfile
from runner.types import SpeechRequest


class EngineAdapter(ABC):
    """Interface that every TTS backend must implement."""

    def __init__(self, profile: ModelProfile, **kwargs: Any) -> None:
        self.profile = profile

    @abstractmethod
    def is_available(self) -> bool:
        """Return ``True`` if this adapter's backend is ready to serve."""
        ...

    @abstractmethod
    def capabilities(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict describing supported features."""
        ...

    @abstractmethod
    def list_voices(self) -> list[dict[str, Any]]:
        """Return available voices for this backend."""
        ...

    @abstractmethod
    def synthesize(self, request: SpeechRequest) -> tuple[bytes, str]:
        """Synthesize audio. Returns ``(audio_bytes, mime_type_or_format)``."""
        ...

    @abstractmethod
    def stream(self, request: SpeechRequest) -> Any:
        """Return a streaming iterator/generator for realtime audio."""
        ...

    @abstractmethod
    def health(self) -> dict[str, Any]:
        """Return a health-check dict for this adapter."""
        ...
