"""
LavaSR audio super-resolution integration for upsampling 24kHz to 48kHz.
Uses LavaSR neural network for high-quality bandwidth extension.
Supports direct 24kHz input (no downsampling needed).
"""

import torch
import torchaudio.transforms as T
import numpy as np
from typing import Optional
from pathlib import Path
import sys

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

from lavasr.enhancer.enhancer import LavaBWE
from lavasr.enhancer.linkwitz_merge import FastLRMerge


class LavaSRUpsampler:
    """
    GPU-accelerated audio upsampler that converts 24kHz audio to 48kHz.
    Uses LavaSR neural network for high-quality super-resolution.
    Direct 24kHz -> 48kHz (no intermediate downsampling).
    """

    def __init__(self, device: str = "cuda", enable: bool = True):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.enabled = enable
        self._upsampler: Optional[T.Resample] = None
        self._bwe_model: Optional[LavaBWE] = None
        self.input_sr = 24000
        self.output_sr = 48000

        if not self.enabled:
            print("[LavaSR] Super-resolution disabled")

    def load(self) -> None:
        if not self.enabled:
            return

        try:
            print(
                f"[LavaSR] Initializing neural upsampler (24kHz -> 48kHz) on {self.device}..."
            )

            self._upsampler = T.Resample(
                orig_freq=self.input_sr,
                new_freq=self.output_sr,
                resampling_method="sinc_interp_kaiser",
            ).to(self.device)

            from huggingface_hub import snapshot_download

            model_path = snapshot_download("YatharthS/LavaSR")

            self._bwe_model = LavaBWE(
                f"{model_path}/enhancer_v2", device=str(self.device)
            )
            self._bwe_model.lr_refiner = FastLRMerge(
                device=str(self.device), cutoff=self.input_sr // 2, transition_bins=1024
            )

            print(f"[LavaSR] Model loaded successfully")

        except Exception as e:
            print(f"[LavaSR] Error during initialization: {e}")
            import traceback

            traceback.print_exc()
            self._bwe_model = None

    def upsample(self, audio: np.ndarray, sample_rate: int = 24000) -> np.ndarray:
        if not self.enabled or self._bwe_model is None:
            return audio

        audio_tensor = torch.from_numpy(audio).float().to(self.device)

        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        with torch.no_grad():
            resampled = self._upsampler(audio_tensor)
            enhanced = self._bwe_model.infer(resampled, autocast=True)

        return enhanced.squeeze(0).cpu().float().numpy()

    def upsample_chunks(
        self, audio_chunk: np.ndarray, sample_rate: int = 24000
    ) -> np.ndarray:
        return self.upsample(audio_chunk, sample_rate)
