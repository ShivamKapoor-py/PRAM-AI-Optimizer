import os


def get_model_size_gb(path):
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024**3)

    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            total += os.path.getsize(fp)

    return total / (1024**3)


def detect_device():
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
    except:
        pass

    return "cpu"


def route(model_path):
    size = get_model_size_gb(model_path)
    device = detect_device()

    print(f"[ROUTER] Model Size: {round(size, 2)} GB")
    print(f"[ROUTER] Device: {device}")

    if size < 1:
        return {
            "pipeline": "pytorch",
            "precision": "fp32",
            "device": device
        }

    elif size < 4:
        return {
            "pipeline": "pytorch",
            "precision": "int8",
            "device": device
        }

    else:
        return {
            "pipeline": "gguf",
            "precision": "q4",
            "device": device
        }