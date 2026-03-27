from __future__ import annotations

import argparse
import time

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keep a CUDA GPU lightly busy to prevent idle shutdown.")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device, e.g. cuda:0")
    parser.add_argument("--matrix-size", type=int, default=4096, help="Square matrix size for matmul workload")
    parser.add_argument("--dtype", type=str, default="float16", choices=("float16", "float32", "bfloat16"))
    parser.add_argument(
        "--reserve-mb",
        type=int,
        default=0,
        help="Extra VRAM to reserve in MiB using a persistent tensor buffer",
    )
    parser.add_argument("--work-seconds", type=float, default=1.5, help="Seconds spent computing per cycle")
    parser.add_argument("--sleep-seconds", type=float, default=2.0, help="Seconds slept between compute cycles")
    parser.add_argument("--log-interval", type=float, default=30.0, help="Heartbeat print interval in seconds")
    return parser.parse_args()


def resolve_dtype(dtype_name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[dtype_name]


def reserve_gpu_memory(device: torch.device, dtype: torch.dtype, reserve_mb: int) -> list[torch.Tensor]:
    if reserve_mb <= 0:
        return []

    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    chunk_mb = 256
    chunk_elements = (chunk_mb * 1024 * 1024) // bytes_per_element
    remaining_mb = reserve_mb
    buffers: list[torch.Tensor] = []
    while remaining_mb > 0:
        current_mb = min(chunk_mb, remaining_mb)
        current_elements = max(1, (current_mb * 1024 * 1024) // bytes_per_element)
        buffers.append(torch.empty((current_elements,), device=device, dtype=dtype))
        remaining_mb -= current_mb
    torch.cuda.synchronize(device)
    return buffers


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError(f"Expected a CUDA device, got: {args.device}")

    torch.cuda.set_device(device)
    dtype = resolve_dtype(args.dtype)

    a = torch.randn((args.matrix_size, args.matrix_size), device=device, dtype=dtype)
    b = torch.randn((args.matrix_size, args.matrix_size), device=device, dtype=dtype)
    c = torch.empty((args.matrix_size, args.matrix_size), device=device, dtype=dtype)
    reserved_buffers = reserve_gpu_memory(device, dtype, args.reserve_mb)

    # Warm up kernels so the steady-state loop is stable.
    for _ in range(4):
        c = torch.matmul(a, b)
    torch.cuda.synchronize(device)

    print(
        "keep_gpu_busy started:",
        f"device={device}",
        f"matrix_size={args.matrix_size}",
        f"dtype={dtype}",
        f"reserve_mb={args.reserve_mb}",
        f"work_seconds={args.work_seconds}",
        f"sleep_seconds={args.sleep_seconds}",
        flush=True,
    )

    last_log_time = 0.0
    iteration = 0
    while True:
        compute_deadline = time.monotonic() + args.work_seconds
        with torch.no_grad():
            while time.monotonic() < compute_deadline:
                c = torch.matmul(a, b)
                a = torch.matmul(c, a)
                b = torch.matmul(b, c)
        torch.cuda.synchronize(device)
        iteration += 1

        now = time.monotonic()
        if now - last_log_time >= args.log_interval:
            mem_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
            print(
                f"[heartbeat] iter={iteration} allocated_mb={mem_mb:.1f} "
                f"reserved_buffers={len(reserved_buffers)} "
                f"work_seconds={args.work_seconds} sleep_seconds={args.sleep_seconds}",
                flush=True,
            )
            last_log_time = now

        if args.sleep_seconds > 0.0:
            time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
