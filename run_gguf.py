from optimizer.gguf import convert_to_gguf
from optimizer.benchmark import benchmark_gguf
import subprocess
import os

HF_MODEL = "models/hf_model"

F16_MODEL = "models/model-f16.gguf"
Q4_MODEL = "models/model-q4.gguf"

DEVICE = "mps"  # or "cpu"


# ------------------------
# STEP 1: CONVERT
# ------------------------
if not os.path.exists(F16_MODEL):
    convert_to_gguf(HF_MODEL, F16_MODEL, quant="f16")


# ------------------------
# STEP 2: QUANTIZE
# ------------------------
if not os.path.exists(Q4_MODEL):
    print("[INFO] Quantizing → Q4...")

    subprocess.run([
        "./llama.cpp/build/bin/llama-quantize",
        F16_MODEL,
        Q4_MODEL,
        "q4_0"
    ], check=True)


# ------------------------
# STEP 3: BENCHMARK
# ------------------------
benchmark_gguf(F16_MODEL, Q4_MODEL, DEVICE)