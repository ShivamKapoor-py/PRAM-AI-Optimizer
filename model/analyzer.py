# model/analyzer.py

import os


def get_file_size_gb(path):
    return round(os.path.getsize(path) / (1024**3), 2)


def get_folder_size_gb(folder_path):
    total = 0
    for root, _, files in os.walk(folder_path):
        for f in files:
            fp = os.path.join(root, f)
            total += os.path.getsize(fp)
    return round(total / (1024**3), 2)


def detect_format_hint(path):
    path_lower = str(path).lower()

    if path_lower.endswith((".pt", ".pth")):
        return "pt"
    elif path_lower.endswith(".gguf"):
        return "gguf"
    elif path_lower.endswith(".safetensors"):
        return "safetensors"

    # Folder-based detection
    if os.path.isdir(path):
        files = []
        for _, _, f in os.walk(path):
            files.extend(f)

        if any(f.endswith(".gguf") for f in files):
            return "gguf"
        elif any(f.endswith(".safetensors") for f in files):
            return "safetensors"
        elif any(f.endswith(".bin") for f in files):
            return "bin"

    return "unknown"


def analyze_model(path):
    if os.path.isfile(path):
        size_gb = get_file_size_gb(path)
        storage_type = "file"

    elif os.path.isdir(path):
        size_gb = get_folder_size_gb(path)
        storage_type = "folder"

    else:
        return {"error": "Invalid path"}

    format_hint = detect_format_hint(path)

    return {
        "model_size_gb": size_gb,
        "storage_type": storage_type,
        "format_hint": format_hint,
        "is_large_model": size_gb is not None and size_gb > 4
    }


if __name__ == "__main__":
    path = input("Enter model path: ").strip()

    from pprint import pprint
    pprint(analyze_model(path))