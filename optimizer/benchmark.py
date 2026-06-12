import os
import subprocess
import time
import re


# ------------------------
# FILE SIZE
# ------------------------
def get_file_size_gb(path):
    return os.path.getsize(path) / (1024**3)


# ------------------------
# EXTRACT GPU MEMORY (REAL)
# ------------------------
def extract_memory(output):
    try:
        # Matches: ( 717 = 606 + 44 + 66 )
        match = re.search(
            r"\(\s*(\d+)\s*=\s*(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)\s*\)",
            output
        )

        if match:
            total = int(match.group(1))
            model = int(match.group(2))
            context = int(match.group(3))
            compute = int(match.group(4))

            return {
                "total_gb": total / 1024,
                "model_gb": model / 1024,
                "context_gb": context / 1024,
                "compute_gb": compute / 1024
            }

    except:
        pass

    return None


# ------------------------
# EXTRACT SPEED
# ------------------------
def extract_speed(output):
    try:
        match = re.search(r"Generation:\s*([\d\.]+)\s*t/s", output)
        if match:
            return float(match.group(1))
    except:
        pass

    return None


# ------------------------
# BENCHMARK
# ------------------------
def benchmark_gguf(f16_path, q4_path, device="mps"):
    print("\n=== GGUF BENCHMARK START ===")

    # ------------------------
    # MODEL SIZE
    # ------------------------
    size_before = get_file_size_gb(f16_path)
    size_after = get_file_size_gb(q4_path)

    # ------------------------
    # RUN MODEL (CAPTURE BOTH STREAMS)
    # ------------------------
    cmd = [
        "./llama.cpp/build/bin/llama-cli",
        "-m", q4_path,
        "-p", "Explain AI simply",
        "-n", "1000"
    ]

    if device == "mps":
        cmd += ["--gpu-layers", "100"]

    t1 = time.time()

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    t2 = time.time()

    output = result.stdout + result.stderr  # 🔥 IMPORTANT

    # ------------------------
    # EXTRACT DATA
    # ------------------------
    memory = extract_memory(output)
    speed = extract_speed(output)

    # ------------------------
    # CALCULATE
    # ------------------------
    reduction = ((size_before - size_after) / size_before) * 100

    # ------------------------
    # PRINT RESULTS
    # ------------------------
    print("\n=== FINAL METRICS ===")

    print(f"Model Size Before (f16): {round(size_before, 3)} GB")
    print(f"Model Size After (Q4):   {round(size_after, 3)} GB")
    print(f"Compression:             {round(reduction, 2)} %")

    if memory:
        print("\n=== GPU MEMORY (REAL) ===")
        print(f"Total Used: {round(memory['total_gb'], 3)} GB")
        print(f"Model:      {round(memory['model_gb'], 3)} GB")
        print(f"Context:    {round(memory['context_gb'], 3)} GB")
        print(f"Compute:    {round(memory['compute_gb'], 3)} GB")
    else:
        print("\n[WARN] Could not extract GPU memory")

    if speed:
        print(f"\nSpeed: {speed} tokens/sec")

    print(f"\nDevice: {device}")
    print(f"Time Taken: {round(t2 - t1, 3)} sec")

    return {
        "size_before": size_before,
        "size_after": size_after,
        "compression": reduction,
        "memory": memory,
        "speed": speed,
        "device": device
    }