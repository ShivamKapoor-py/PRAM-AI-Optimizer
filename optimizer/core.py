import torch
import gc
import platform

# ------------------------
# SET QUANTIZATION ENGINE (CRITICAL)
# ------------------------
if torch.backends.quantized.supported_engines:
    if "qnnpack" in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = "qnnpack"
    elif "fbgemm" in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = "fbgemm"


# ------------------------
# DEVICE DETECTION
# ------------------------
def get_best_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ------------------------
# LOAD MODEL (SAFE)
# ------------------------
def load_model(path):
    obj = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    if hasattr(obj, "eval"):
        obj.eval()
        return obj
    else:
        raise ValueError("Only full PyTorch models supported")


# ------------------------
# MODEL SIZE (REAL METRIC)
# ------------------------
def get_model_size_gb(model):
    total = 0

    for name, param in model.named_parameters():
        try:
            total += param.numel() * param.element_size()
        except:
            pass

    # 🔥 Handle quantized weights (critical fix)
    for module in model.modules():
        if hasattr(module, "_packed_params"):
            try:
                w, _ = module._packed_params._weight_bias()
                total += w.numel() * w.element_size()
            except:
                pass

    return round(total / (1024**3), 6)

# ------------------------
# CPU INT8
# ------------------------
def optimize_cpu(model):
    print("[INFO] Using CPU INT8 optimization")

    try:
        model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        return model
    except Exception as e:
        print("[WARN] INT8 failed, fallback FP32:", e)
        return model


# ------------------------
# MPS FP16
# ------------------------
def optimize_mps(model):
    print("[INFO] Using MPS FP16 optimization")

    model_fp16 = model.half()

    del model
    gc.collect()

    model_fp16 = model_fp16.to("mps")
    torch.mps.empty_cache()

    return model_fp16


# ------------------------
# CHUNKING
# ------------------------
def chunk_model(model, chunk_size=2):
    layers = list(model.children())

    if not layers:
        return [model]

    chunks = []
    for i in range(0, len(layers), chunk_size):
        chunk = torch.nn.Sequential(*layers[i:i + chunk_size])
        chunks.append(chunk)

    return chunks

def run_chunked_forward(chunks, x, device):
    """Run model in chunks to reduce peak memory usage"""
    
    for chunk in chunks:
        chunk = chunk.to(device)

        x = chunk(x)

        # 🔥 Move chunk back to CPU to free memory
        chunk.to("cpu")

        if device == "mps":
            import torch
            torch.mps.empty_cache()

    return x


# ------------------------
# MAIN OPTIMIZER
# ------------------------
def optimize(model_path, strategy=None):
    if strategy is None:
        strategy = {}

    device = strategy.get("device") or get_best_device()
    use_chunking = strategy.get("use_chunking", True)

    print(f"[INFO] Device: {device}")
    print(f"[INFO] System: {platform.system()}")

    model = load_model(model_path)

    size_before = get_model_size_gb(model)

    # 🔹 Optimize
    if device == "mps":
        model = optimize_mps(model)
        precision = "fp16"
    else:
        model = optimize_cpu(model)
        precision = "int8"

    size_after = get_model_size_gb(model)

    # 🔹 Chunking
    if use_chunking:
        chunks = chunk_model(model)
    else:
        chunks = [model]

    return {
        "chunks": chunks,
        "model_size_before": size_before,
        "model_size_after": size_after,
        "precision": precision,
        "device": device
    }