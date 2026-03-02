# 🚀 VibeVoice Realtime Optimization Plan (Ampere GPUs)

## 🎯 Goal
Reduce the Real-Time Factor (RTF) from **0.335x** to **<0.28x** by leveraging Ampere-specific features and reducing CPU-GPU synchronization overhead.

---

## ✅ Implementation Status

| # | Optimization | Status | Notes |
|---|--------------|--------|-------|
| 1️⃣ | `torch.compile()` | ✅ Done | `app.py:153-167` |
| 2️⃣ | GPU Upsampling | ✅ Done | LavaSR neural upsampler (see below) |
| 3️⃣ | CUDA Stream Pipelining | ✅ Done | `app.py:71-72, 88-89, 278-295, 403-411` |

---

## 🆕 Optimization 2: LavaSR Neural Upsampler (Final)

LavaSR is a 50MB neural network that provides superior quality and speed for audio super-resolution.

### Why LavaSR?
- **Direct 24kHz → 48kHz** (no intermediate downsampling like NovaSR)
- **Speed**: ~1700x realtime
- **Quality**: Surpasses 6GB diffusion models (best LSD scores)
- **Flexible input**: Supports 8-48kHz input rates

### Implementation: `overrides/lavasr_upsampler.py`
```python
from overrides.lavasr_upsampler import LavaSRUpsampler

# Direct 24kHz -> 48kHz neural super-resolution
upsampler = LavaSRUpsampler(device="cuda", enable=True)
upsampler.load()
upsampled = upsampler.upsample(audio_24kHz)  # Returns 48kHz enhanced audio
```

### Benchmark Results (RTX 3090)
| Input Duration | Upsampling Time | RTF | Speed |
|----------------|-----------------|-----|-------|
| 1.0s | ~3ms | 0.003x | 333x realtime |
| 5.0s | ~3ms | 0.0006x | 1725x realtime |
| 10.0s | ~6ms | 0.0006x | 1666x realtime |

### Comparison
| Model | Speed | Quality (LSD) | Input Support |
|-------|-------|---------------|---------------|
| **LavaSR** | 5000x realtime | 0.63 (24→48kHz) | 8-48kHz |
| NovaSR | 3600x realtime | ~0.72 (16→48kHz) | 16kHz only |
| FlashSR | 14x realtime | - | - |
| AudioSR | 0.6x realtime | 0.82 (24→48kHz) | - |

---

## 1️⃣ Optimization: Kernel Fusion with `torch.compile()`
PyTorch 2.0+ `torch.compile()` fusions kernels and reduces Python runtime overhead. For Ampere GPUs (RTX 3090), the `reduce-overhead` mode uses CUDA Graphs to eliminate kernel launch bottlenecks.

### Target: `overrides/app.py`
```python
if self.device == "cuda" and hasattr(torch, 'compile'):
    self.model = torch.compile(
        self.model, 
        mode="reduce-overhead",
        fullgraph=False
    )
```

---

## 3️⃣ Optimization: CUDA Stream Pipelining
Dual CUDA streams to overlap computation with memory transfer.

### Target: `overrides/app.py`
```python
# In __init__ and load()
self._compute_stream = torch.cuda.Stream()
self._transfer_stream = torch.cuda.Stream()

# In _run_generation()
with torch.cuda.stream(self._compute_stream):
    self.model.generate(...)

# In stream() loop for GPU -> CPU transfer
with torch.cuda.stream(self._transfer_stream):
    audio_chunk = audio_chunk.to("cpu", non_blocking=True)
```

---

## 📈 Expected Impact
| Metric | Baseline | Post-Optimization |
|--------|----------|-------------------|
| **RTF** | 0.335x | **<0.28x** |
| **Speed** | 3.0x Realtime | **3.5x+ Realtime** |
| **Upsampling Latency** | ~5-10ms | **<3ms** |

---

## 🛠️ Verification
1. **Benchmark upsampler**: `python test_lavasr.py`
2. **Full TTS benchmark**: `python benchmark.py --device cuda:0`
3. **Quality check**: Verify audio has no artifacts
