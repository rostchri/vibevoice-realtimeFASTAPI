"""
NovaSR audio super-resolution integration for upsampling 24kHz to 48kHz.
Uses NovaSR neural network (16kHz -> 48kHz) with pre-downsampling from 24kHz.
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio.transforms as T

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

from novasr import FastSR


class NovaSRUpsampler:
    """
    GPU-accelerated audio upsampler that converts 24kHz audio to 48kHz.
    Uses NovaSR neural network for high-quality super-resolution.
    Pipeline: 24kHz -> 16kHz (downsample) -> NovaSR -> 48kHz
    """

    def __init__(self, device: str = "cuda", enable: bool = True):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.enabled = enable
        self._downsampler: Optional[T.Resample] = None
        self._novasr: Optional[FastSR] = None
        self.input_sr = 24000
        self.novasr_input_sr = 16000
        self.output_sr = 48000

        if not self.enabled:
            print("[NovaSR] Super-resolution disabled")

    def load(self) -> None:
        if not self.enabled:
            return

        try:
            print(
                f"[NovaSR] Initializing neural upsampler (24kHz -> 16kHz -> 48kHz) on {self.device}..."
            )

            self._downsampler = T.Resample(
                orig_freq=self.input_sr,
                new_freq=self.novasr_input_sr,
                resampling_method="sinc_interp_kaiser",
            ).to(self.device)

            use_half = self.device.type == "cuda"
            self._novasr = FastSR(half=use_half)
            self._novasr.device = self.device
            self._novasr.model = self._novasr.model.to(self.device)

            print(f"[NovaSR] Model loaded successfully (half precision: {use_half})")

        except Exception as e:
            print(f"[NovaSR] Error during initialization: {e}")
            import traceback

            traceback.print_exc()
            self._novasr = None

    def upsample(self, audio: np.ndarray, sample_rate: int = 24000) -> np.ndarray:
        if not self.enabled or self._novasr is None:
            return audio

        audio_tensor = torch.from_numpy(audio).float().to(self.device)

        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        with torch.no_grad():
            downsampled = self._downsampler(audio_tensor)

            novasr_input = downsampled.unsqueeze(1)
            if self._novasr.half:
                novasr_input = novasr_input.half()

            upsampled = self._novasr.infer(novasr_input)

        if upsampled.ndim == 2:
            upsampled = upsampled.squeeze(0)

        return upsampled.cpu().float().numpy()

    def upsample_chunks(self, audio_chunk: np.ndarray, sample_rate: int = 24000) -> np.ndarray:
        return self.upsample(audio_chunk, sample_rate)
