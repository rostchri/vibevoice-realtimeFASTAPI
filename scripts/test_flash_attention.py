#!/usr/bin/env python3
"""Test Flash Attention 2 installation and functionality."""

import sys

import torch


def test_flash_attention():
    print("=" * 60)
    print("Flash Attention 2 Installation Test")
    print("=" * 60)

    print("\n[1/5] Checking CUDA availability...")
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return False

    device_name = torch.cuda.get_device_name(0)
    compute_cap = torch.cuda.get_device_capability(0)
    print(f"✅ CUDA available: {device_name}")
    print(f"   Compute capability: {compute_cap}")
    print(f"   CUDA version: {torch.version.cuda}")

    print("\n[2/5] Importing flash_attn...")
    try:
        import flash_attn

        print("✅ flash_attn imported successfully")
        print(f"   Version: {flash_attn.__version__}")
    except ImportError as e:
        print(f"❌ Failed to import flash_attn: {e}")
        return False

    print("\n[3/5] Checking Flash Attention 2 availability...")
    try:
        from flash_attn import flash_attn_func

        print("✅ flash_attn_func available")
    except ImportError as e:
        print(f"❌ flash_attn_func not available: {e}")
        return False

    print("\n[4/5] Running basic functionality test...")
    try:
        batch_size = 2
        seq_len = 128
        num_heads = 8
        head_dim = 64

        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16
        )
        k = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16
        )
        v = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16
        )

        output = flash_attn_func(q, k, v, softmax_scale=1.0 / head_dim**0.5)
        print("✅ Flash Attention 2 functional test passed")
        print(f"   Input shape: {q.shape}")
        print(f"   Output shape: {output.shape}")
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

    print("\n[5/5] Checking GPU memory...")
    mem_allocated = torch.cuda.memory_allocated(0) / 1024**2
    mem_reserved = torch.cuda.memory_reserved(0) / 1024**2
    print(f"✅ Memory allocated: {mem_allocated:.2f} MB")
    print(f"   Memory reserved: {mem_reserved:.2f} MB")

    print("\n" + "=" * 60)
    print("✅ All Flash Attention 2 tests passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_flash_attention()
    sys.exit(0 if success else 1)
