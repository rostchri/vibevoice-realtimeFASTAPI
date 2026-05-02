"""Longform adapter – pluggable scaffold for 1.5B / 7B models.

The adapter checks at runtime whether a usable long-form backend is available.
If it is not, every operation returns a clear :class:`BackendUnavailableError`
rather than crashing or silently falling back to the realtime path.

When a real long-form inference backend is wired in, replace the TODO stubs in
:meth:`_load_backend`, :meth:`synthesize`, etc.
"""

from __future__ import annotations

from threading import Lock
from typing import Any, ClassVar

from runner.adapters.base import EngineAdapter
from runner.errors import BackendUnavailableError, CapabilityError
from runner.model_registry import ModelProfile
from runner.types import SpeechRequest


class LongformNativeAdapter(EngineAdapter):
    """Adapter for VibeVoice 1.5B / 7B long-form models."""

    _backend_status_cache: ClassVar[dict[str, tuple[bool, str | None]]] = {}
    _backend_status_cache_lock: ClassVar[Lock] = Lock()

    def __init__(self, profile: ModelProfile, **kwargs: Any) -> None:
        super().__init__(profile, **kwargs)
        self._backend_loaded = False
        self._backend_error: str | None = None

    # ------------------------------------------------------------------
    # Backend detection
    # ------------------------------------------------------------------

    def _load_backend(self) -> bool:
        """Attempt to import and initialise the long-form backend.

        Returns ``True`` on success, ``False`` otherwise.

        .. note::
            Imports are done *inside* this method to avoid hard top-level
            dependencies that would break installations where the longform
            libraries are not present.

        TODO: Replace the try/except block below with actual backend loading
              once a compatible long-form inference module is available, e.g.::

                  from vibevoice.modeling_vibevoice import (
                      VibeVoiceForConditionalGeneration,
                  )
        """
        try:
            # TODO: Replace with real long-form backend import when available.
            # This ImportError is intentional – no long-form backend exists yet.
            raise ImportError("No long-form backend available yet – this is expected")
        except ImportError as exc:
            self._backend_error = str(exc)
            return False

    def _ensure_backend(self) -> None:
        """Raise :class:`BackendUnavailableError` if backend is missing."""
        with self._backend_status_cache_lock:
            cached = self._backend_status_cache.get(self.profile.key)
        if cached is not None:
            available, error = cached
            self._backend_error = error
            if available:
                self._backend_loaded = True
                return
            raise BackendUnavailableError(self.profile.key, detail=self._backend_error)

        if self._backend_loaded:
            return

        available = self._load_backend()
        with self._backend_status_cache_lock:
            self._backend_status_cache[self.profile.key] = (
                available,
                None if available else self._backend_error,
            )

        if not available:
            raise BackendUnavailableError(
                self.profile.key,
                detail=self._backend_error,
            )

        self._backend_loaded = True

    # ------------------------------------------------------------------
    # EngineAdapter interface
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        try:
            self._ensure_backend()
            return True
        except BackendUnavailableError:
            return False

    def get_backend_error(self) -> str | None:
        """Return the last backend loading error message, or ``None``."""
        return self._backend_error

    def capabilities(self) -> dict[str, Any]:
        available = self.is_available()
        return {
            "model": self.profile.key,
            "family": self.profile.family,
            "supports_stream": self.profile.supports_stream,
            "supports_multispeaker": self.profile.supports_multispeaker,
            "supports_voice_list": self.profile.supports_voice_list,
            "status": "available" if available else "backend_unavailable",
        }

    def list_voices(self) -> list[dict[str, Any]]:
        # Longform models do not ship preset voice lists in the current tree.
        return []

    def synthesize(self, request: SpeechRequest) -> tuple[bytes, str]:
        """Synthesize audio from text or multi-speaker dialogue.

        TODO: Wire in real inference once a long-form backend is available.
        """
        self._ensure_backend()
        # Placeholder – real implementation goes here.
        raise NotImplementedError("Long-form synthesis not yet implemented.")

    def stream(self, request: SpeechRequest) -> Any:
        """Long-form models do **not** support WebSocket streaming."""
        raise CapabilityError(self.profile.key, "stream")

    def health(self) -> dict[str, Any]:
        available = self.is_available()
        result: dict[str, Any] = {
            "adapter": "longform_native",
            "model": self.profile.key,
            "family": self.profile.family,
            "available": available,
        }
        if not available:
            result["error"] = self._backend_error
        return result
