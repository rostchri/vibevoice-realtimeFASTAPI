"""Tests for the runner package – model registry, types, adapters, and errors.

These tests validate the multi-model infrastructure without requiring
GPU hardware or model weights.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure runner package is importable
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from runner.adapter_factory import make_adapter  # noqa: E402
from runner.adapters.longform_native import LongformNativeAdapter  # noqa: E402
from runner.adapters.realtime_demo import RealtimeDemoAdapter  # noqa: E402
from runner.errors import (  # noqa: E402
    BackendUnavailableError,
    CapabilityError,
    InvalidRequestForModelError,
    UnknownModelError,
)
from runner.model_registry import (  # noqa: E402
    DEFAULT_MODEL_KEY,
    get_model_profile,
    list_aliases,
    list_model_keys,
    resolve_model_key,
)
from runner.types import (  # noqa: E402
    SpeechRequest,
    validate_for_longform,
    validate_for_realtime,
)

# =====================================================================
# Model registry
# =====================================================================


class TestModelRegistry:
    def test_default_model_key(self) -> None:
        assert DEFAULT_MODEL_KEY == "realtime-0.5b"

    def test_list_model_keys(self) -> None:
        keys = list_model_keys()
        assert "realtime-0.5b" in keys
        assert "tts-1.5b" in keys
        assert "tts-7b" in keys

    def test_list_aliases(self) -> None:
        aliases = list_aliases()
        assert aliases["tts-1"] == "realtime-0.5b"
        assert aliases["tts-1-hd"] == "realtime-0.5b"
        assert aliases["vibevoice-realtime-0.5b"] == "realtime-0.5b"
        assert aliases["vibevoice-1.5b"] == "tts-1.5b"
        assert aliases["vibevoice-7b"] == "tts-7b"

    def test_get_profile_realtime(self) -> None:
        p = get_model_profile("realtime-0.5b")
        assert p.family == "realtime"
        assert p.loader_mode == "subprocess_demo"
        assert p.supports_stream is True
        assert p.supports_multispeaker is False
        assert p.hf_model_id == "microsoft/VibeVoice-Realtime-0.5B"

    def test_get_profile_longform_1_5b(self) -> None:
        p = get_model_profile("tts-1.5b")
        assert p.family == "longform"
        assert p.loader_mode == "native_longform"
        assert p.supports_stream is False
        assert p.supports_multispeaker is True

    def test_get_profile_longform_7b(self) -> None:
        p = get_model_profile("tts-7b")
        assert p.family == "longform"
        assert p.supports_multispeaker is True

    def test_get_profile_unknown_raises(self) -> None:
        with pytest.raises(UnknownModelError):
            get_model_profile("nonexistent-model")


# =====================================================================
# Alias resolution
# =====================================================================


class TestAliasResolution:
    def test_none_defaults(self) -> None:
        assert resolve_model_key(None) == "realtime-0.5b"

    def test_empty_string_defaults(self) -> None:
        assert resolve_model_key("") == "realtime-0.5b"

    def test_whitespace_defaults(self) -> None:
        assert resolve_model_key("   ") == "realtime-0.5b"

    def test_canonical_key(self) -> None:
        assert resolve_model_key("realtime-0.5b") == "realtime-0.5b"
        assert resolve_model_key("tts-1.5b") == "tts-1.5b"
        assert resolve_model_key("tts-7b") == "tts-7b"

    def test_alias_tts_1(self) -> None:
        assert resolve_model_key("tts-1") == "realtime-0.5b"

    def test_alias_tts_1_hd(self) -> None:
        assert resolve_model_key("tts-1-hd") == "realtime-0.5b"

    def test_alias_vibevoice_realtime(self) -> None:
        assert resolve_model_key("vibevoice-realtime-0.5b") == "realtime-0.5b"

    def test_alias_vibevoice_1_5b(self) -> None:
        assert resolve_model_key("vibevoice-1.5b") == "tts-1.5b"

    def test_alias_vibevoice_7b(self) -> None:
        assert resolve_model_key("vibevoice-7b") == "tts-7b"

    def test_case_insensitive(self) -> None:
        assert resolve_model_key("TTS-1") == "realtime-0.5b"
        assert resolve_model_key("Realtime-0.5B") == "realtime-0.5b"

    def test_unknown_raises(self) -> None:
        with pytest.raises(UnknownModelError):
            resolve_model_key("gpt-4")


# =====================================================================
# Request validation
# =====================================================================


class TestRequestValidation:
    def test_realtime_valid(self) -> None:
        req = SpeechRequest(model="tts-1", input="Hello world")
        assert validate_for_realtime(req) == []

    def test_realtime_missing_input(self) -> None:
        req = SpeechRequest(model="tts-1")
        errors = validate_for_realtime(req)
        assert any("input" in e for e in errors)

    def test_realtime_rejects_speakers(self) -> None:
        from runner.types import SpeakerTurn

        req = SpeechRequest(
            model="tts-1",
            input="Hello",
            speakers=[SpeakerTurn(speaker="Alice", text="Hi")],
        )
        errors = validate_for_realtime(req)
        assert any("speakers" in e for e in errors)

    def test_longform_valid_input(self) -> None:
        req = SpeechRequest(model="tts-1.5b", input="Hello world")
        assert validate_for_longform(req) == []

    def test_longform_valid_speakers(self) -> None:
        from runner.types import SpeakerTurn

        req = SpeechRequest(
            model="tts-1.5b",
            speakers=[SpeakerTurn(speaker="Alice", text="Hi")],
        )
        assert validate_for_longform(req) == []

    def test_longform_missing_both(self) -> None:
        req = SpeechRequest(model="tts-1.5b")
        errors = validate_for_longform(req)
        assert any("input" in e or "speakers" in e for e in errors)

    def test_longform_rejects_stream(self) -> None:
        req = SpeechRequest(model="tts-1.5b", input="Hello", stream=True)
        errors = validate_for_longform(req)
        assert any("stream" in e.lower() for e in errors)

    def test_empty_string_normalised_to_none(self) -> None:
        req = SpeechRequest(input="", voice="")
        assert req.input is None
        assert req.voice is None

    def test_empty_speakers_normalised_to_none(self) -> None:
        req = SpeechRequest(input="Hello", speakers=[])
        assert req.speakers is None


# =====================================================================
# Adapter factory
# =====================================================================


class TestAdapterFactory:
    def test_realtime_adapter(self) -> None:
        adapter = make_adapter("realtime-0.5b")
        assert isinstance(adapter, RealtimeDemoAdapter)
        assert adapter.is_available() is True

    def test_longform_1_5b_adapter(self) -> None:
        adapter = make_adapter("tts-1.5b")
        assert isinstance(adapter, LongformNativeAdapter)

    def test_longform_7b_adapter(self) -> None:
        adapter = make_adapter("tts-7b")
        assert isinstance(adapter, LongformNativeAdapter)

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(UnknownModelError):
            make_adapter("nonexistent-model")


# =====================================================================
# Longform adapter – graceful degradation
# =====================================================================


class TestLongformAdapter:
    def test_is_available_false(self) -> None:
        adapter = make_adapter("tts-1.5b")
        assert adapter.is_available() is False

    def test_capabilities_show_unavailable(self) -> None:
        adapter = make_adapter("tts-1.5b")
        caps = adapter.capabilities()
        assert caps["status"] == "backend_unavailable"
        assert caps["supports_stream"] is False
        assert caps["supports_multispeaker"] is True

    def test_synthesize_raises_backend_unavailable(self) -> None:
        adapter = make_adapter("tts-1.5b")
        req = SpeechRequest(model="tts-1.5b", input="Hello")
        with pytest.raises(BackendUnavailableError):
            adapter.synthesize(req)

    def test_stream_raises_capability_error(self) -> None:
        adapter = make_adapter("tts-1.5b")
        req = SpeechRequest(model="tts-1.5b", input="Hello")
        with pytest.raises(CapabilityError):
            adapter.stream(req)

    def test_health_reports_error(self) -> None:
        adapter = make_adapter("tts-1.5b")
        h = adapter.health()
        assert h["available"] is False
        assert "error" in h


# =====================================================================
# Realtime adapter basics
# =====================================================================


class TestRealtimeAdapter:
    def test_capabilities(self) -> None:
        adapter = make_adapter("realtime-0.5b")
        caps = adapter.capabilities()
        assert caps["status"] == "available"
        assert caps["supports_stream"] is True
        assert caps["family"] == "realtime"

    def test_health(self) -> None:
        adapter = make_adapter("realtime-0.5b")
        h = adapter.health()
        assert h["available"] is True
        assert h["adapter"] == "realtime_demo"


# =====================================================================
# Error classes
# =====================================================================


class TestErrors:
    def test_unknown_model_error(self) -> None:
        err = UnknownModelError("foo")
        assert "foo" in str(err)
        assert err.model_key == "foo"

    def test_capability_error(self) -> None:
        err = CapabilityError("tts-1.5b", "stream")
        assert "stream" in str(err)
        assert err.model_key == "tts-1.5b"

    def test_backend_unavailable_error(self) -> None:
        err = BackendUnavailableError("tts-1.5b")
        assert "tts-1.5b" in str(err)
        assert err.model_key == "tts-1.5b"

    def test_backend_unavailable_with_detail(self) -> None:
        err = BackendUnavailableError("tts-7b", detail="Install vibevoice-longform")
        assert "Install vibevoice-longform" in str(err)

    def test_invalid_request_error(self) -> None:
        err = InvalidRequestForModelError("bad field")
        assert "bad field" in str(err)


# =====================================================================
# Backward compatibility – download_model default resolution
# =====================================================================


class TestDownloaderDefaults:
    """Ensure the downloader resolves defaults correctly from the registry."""

    def test_default_model_resolves(self) -> None:
        key = resolve_model_key("realtime-0.5b")
        profile = get_model_profile(key)
        assert profile.hf_model_id == "microsoft/VibeVoice-Realtime-0.5B"
        assert profile.default_local_dir == "models/VibeVoice-Realtime-0.5B"

    def test_1_5b_model_resolves(self) -> None:
        key = resolve_model_key("tts-1.5b")
        profile = get_model_profile(key)
        assert profile.hf_model_id == "microsoft/VibeVoice-1.5B"
        assert profile.default_local_dir == "models/VibeVoice-1.5B"

    def test_7b_model_resolves(self) -> None:
        key = resolve_model_key("tts-7b")
        profile = get_model_profile(key)
        assert "7B" in profile.default_local_dir
