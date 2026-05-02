"""Shared request types for the runner package."""

from __future__ import annotations

from pydantic import BaseModel, field_validator, model_validator


class SpeakerTurn(BaseModel):
    """A single speaker turn for multi-speaker dialogue (longform models)."""

    speaker: str
    text: str


class SpeechRequest(BaseModel):
    """Unified speech-synthesis request accepted by all adapters."""

    model: str | None = None
    input: str | None = None
    voice: str | None = None
    response_format: str = "opus"
    temp: float | None = None
    speed: float | None = None
    stream: bool = False
    speakers: list[SpeakerTurn] | None = None

    @field_validator("input", mode="before")
    @classmethod
    def _normalize_empty_input(cls, v: str | None) -> str | None:
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    @field_validator("voice", mode="before")
    @classmethod
    def _normalize_empty_voice(cls, v: str | None) -> str | None:
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    @model_validator(mode="after")
    def _normalize_empty_speakers(self) -> "SpeechRequest":
        if self.speakers is not None and len(self.speakers) == 0:
            self.speakers = None
        return self


def validate_for_realtime(req: SpeechRequest) -> list[str]:
    """Return a list of validation errors for a realtime-family request."""
    errors: list[str] = []
    if not req.input:
        errors.append("Field 'input' is required for realtime models.")
    if req.speakers:
        errors.append("Field 'speakers' is not supported by realtime models.")
    return errors


def validate_for_longform(req: SpeechRequest) -> list[str]:
    """Return a list of validation errors for a longform-family request."""
    errors: list[str] = []
    if not req.input and not req.speakers:
        errors.append("Either 'input' or 'speakers' is required for longform models.")
    if req.stream:
        errors.append("Streaming is not supported by longform models.")
    return errors
