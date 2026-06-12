import os
import subprocess


def convert_to_gguf(hf_model_path, output_path, quant="f16"):
    script_path = "llama.cpp/convert_hf_to_gguf.py"

    if not os.path.exists(script_path):
        raise RuntimeError("convert_hf_to_gguf.py not found")

    cmd = [
        "python",
        script_path,
        hf_model_path,
        "--outfile", output_path,
        "--outtype", quant
    ]

    print("[INFO] Converting → GGUF (f16)...")
    subprocess.run(cmd, check=True)

    return output_path