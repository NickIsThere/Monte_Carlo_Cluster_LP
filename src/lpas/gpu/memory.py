from __future__ import annotations


def estimate_batch_memory_bytes(K: int, n: int, m: int, *, num_buffers: int = 6, dtype_bytes: int = 4) -> int:
    if K <= 0 or n <= 0 or m <= 0:
        raise ValueError("K, n, and m must all be positive")
    if num_buffers <= 0 or dtype_bytes <= 0:
        raise ValueError("num_buffers and dtype_bytes must be positive")
    return int(K * (n + m) * num_buffers * dtype_bytes)
