from optimizer.router import route
from optimizer.core import load_model
import subprocess
import os

MODEL_PATH = "models/hf_model"


decision = route(MODEL_PATH)


# ------------------------
# GGUF PIPELINE
# ------------------------
if decision["pipeline"] == "gguf":
    print("\n[PIPELINE] GGUF")

    from optimizer.gguf import convert_to_gguf
    from optimizer.benchmark import benchmark_gguf

    F16 = "models/model-f16.gguf"
    Q4 = "models/model-q4.gguf"

    if not os.path.exists(F16):
        convert_to_gguf(MODEL_PATH, F16, "f16")

    if not os.path.exists(Q4):
        subprocess.run([
            "./llama.cpp/build/bin/llama-quantize",
            F16,
            Q4,
            "q4_0"
        ], check=True)

    benchmark_gguf(F16, Q4, decision["device"])


# ------------------------
# PYTORCH PIPELINE
# ------------------------
elif decision["pipeline"] == "pytorch":
    print("\n[PIPELINE] PYTORCH")

    import torch
    from optimizer.disk_splitter import split_model, run_chunked_forward

    model = load_model("real_model.pt")

    if decision["precision"] == "int8":
        model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )

    device = decision["device"]

    chunks = split_model(model, num_chunks=2)

    dummy = torch.randn(1, 512)

    output = run_chunked_forward(chunks, dummy, device)

    print("[SUCCESS] Output shape:", output.shape)