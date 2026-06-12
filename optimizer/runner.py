import subprocess


# ------------------------
# CPU RUN (MMAP DEFAULT)
# ------------------------
def run_cpu(model_path, prompt):
    cmd = [
        "./llama.cpp/main",
        "-m", model_path,
        "-p", prompt,
        "--n-predict", "50"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout


# ------------------------
# METAL RUN (MPS)
# ------------------------
def run_metal(model_path, prompt):
    cmd = [
        "./llama.cpp/main",
        "-m", model_path,
        "-p", prompt,
        "--n-predict", "50",
        "--gpu-layers", "100"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout


# ------------------------
# CHUNKING (REAL)
# ------------------------
def chunk_prompt(prompt, chunk_size=50):
    words = prompt.split()
    return [
        " ".join(words[i:i+chunk_size])
        for i in range(0, len(words), chunk_size)
    ]


def run_chunked(model_path, prompt, device="cpu"):
    chunks = chunk_prompt(prompt)
    outputs = []

    for chunk in chunks:
        if device == "mps":
            out = run_metal(model_path, chunk)
        else:
            out = run_cpu(model_path, chunk)

        outputs.append(out)

    return " ".join(outputs)