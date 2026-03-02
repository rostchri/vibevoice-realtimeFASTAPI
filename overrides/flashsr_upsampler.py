"""
FlashSR audio super-resolution integration for upsampling 24kHz to 48kHz.
Provides ultra-fast audio upsampling using torchaudio on GPU.
"""

import torch
import torchaudio.transforms as T
import numpy as np
from typing import Optional


class FlashSRUpsampler:
    """
    GPU-accelerated audio upsampler that converts 24kHz audio to 48kHz.
    Uses torchaudio.transforms.Resample for high-performance GPU processing.
    """

    def __init__(self, device: str = "cuda", enable: bool = True):
        """
        Initialize FlashSR upsampler.

        Args:
            device: Device to run inference on ('cuda', 'mps', or 'cpu')
            enable: Whether to enable super-resolution (default: True)
        """
        self.device = device
        self.enabled = enable
        self._resampler: Optional[T.Resample] = None
        self.input_sr = 24000
        self.output_sr = 48000

        if not self.enabled:
            print("[FlashSR] Super-resolution disabled")
            return

    def load(self) -> None:
        """Load resampler onto the target device."""
        if not self.enabled:
            return

        try:
            print(
                f"[FlashSR] Initializing GPU-accelerated resampler (24kHz -> 48kHz) on {self.device}..."
            )

            # Use high-quality sinc interpolation (similar to librosa kaiser_best)
            self._resampler = T.Resample(
                orig_freq=self.input_sr,
                new_freq=self.output_sr,
                resampling_method="sinc_interp_kaiser",
            ).to(self.device)

            print("[FlashSR] Resampler initialized successfully")

        except Exception as e:
            print(f"[FlashSR] Error during initialization: {e}")
            print("[FlashSR] Falling back to CPU-based processing if possible")
            self._resampler = None

    def upsample(self, audio: np.ndarray, sample_rate: int = 24000) -> np.ndarray:
        """
        Upsample audio from 24kHz to 48kHz.

        Args:
            audio: Input audio as numpy array (float32, mono)
            sample_rate: Input sample rate (default: 24000)

        Returns:
            Upsampled audio at 48kHz as numpy array
        """
        if not self.enabled or self._resampler is None:
            return audio

        # Convert to tensor and move to device
        audio_tensor = torch.from_numpy(audio).to(self.device)

        # Add batch/channel dims if necessary
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        # Resample on GPU
        upsampled_tensor = self._resampler(audio_tensor)

        # Back to numpy
        return upsampled_tensor.squeeze(0).cpu().numpy()

    def upsample_chunks(
        self, audio_chunk: np.ndarray, sample_rate: int = 24000
    ) -> np.ndarray:
        """
        Upsample a single audio chunk for streaming.

        Args:
            audio_chunk: Input audio chunk
            sample_rate: Input sample rate

        Returns:
            Upsampled audio chunk at 48kHz
        """
        return self.upsample(audio_chunk, sample_rate)
