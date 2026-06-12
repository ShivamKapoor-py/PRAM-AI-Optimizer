# hardware/detect.py

import torch
import psutil
import platform
import subprocess
import re


def get_apple_chip_info():
    """Detect Apple Silicon chip and generation"""
    try:
        output = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"]
        ).decode()

        # Extract M number (M1, M2, M3...)
        match = re.search(r"M(\d+)", output)
        if match:
            generation = int(match.group(1))
            chip_name = f"M{generation}"
        else:
            generation = 0
            chip_name = "Apple Silicon"

        return {
            "chip_name": chip_name,
            "generation": generation
        }

    except:
        return {
            "chip_name": "Unknown",
            "generation": 0
        }


def detect_hardware():
    system = platform.system()

    # CPU
    cpu_cores = psutil.cpu_count(logical=True)

    # RAM
    memory = psutil.virtual_memory()
    ram_total_gb = round(memory.total / (1024**3), 2)
    ram_available_gb = round(memory.available / (1024**3), 2)
    memory_usage_percent = memory.percent

    # Defaults
    device = "cpu"
    mps_available = False
    unified_memory = False
    chip_info = {"chip_name": None, "generation": 0}

    if system == "Darwin":
        chip_info = get_apple_chip_info()
        unified_memory = True

        if torch.backends.mps.is_available():
            mps_available = True
            device = "mps"

    return {
        "system": system,
        "chip": chip_info["chip_name"],
        "chip_generation": chip_info["generation"],
        "device": device,
        "cpu_cores": cpu_cores,
        "ram_total_gb": ram_total_gb,
        "ram_available_gb": ram_available_gb,
        "memory_usage_percent": memory_usage_percent,
        "mps_available": mps_available,
        "unified_memory": unified_memory
    }


# 🔥 DECISION ENGINE (V1)
def decide_optimization_strategy(hardware_info, model_size_gb):
    gen = hardware_info["chip_generation"]
    ram_free = hardware_info["ram_available_gb"]
    mem_pressure = hardware_info["memory_usage_percent"]

    strategy = {
        "precision": None,
        "use_chunking": False,
        "use_mmap": False,
        "use_gguf": False,
        "device": hardware_info["device"],
        "notes": []
    }

    # 🚨 HIGH MEMORY PRESSURE
    if mem_pressure > 75 or ram_free < 2:
        strategy["use_chunking"] = True
        strategy["use_mmap"] = True
        strategy["notes"].append("High memory pressure → aggressive optimization")

    # 🔹 SMALL MODELS (<4GB)
    if model_size_gb <= 4:
        if gen <= 2:  # M1, M2
            strategy["precision"] = "fp16"
            strategy["use_chunking"] = True
            strategy["notes"].append("Older chip → chunked FP16")

        elif gen >= 3:  # M3, M4, M5+
            strategy["precision"] = "fp16"
            strategy["notes"].append("New chip → full FP16 allowed")

    # 🔹 LARGE MODELS (>4GB)
    else:
        strategy["use_gguf"] = True
        strategy["use_mmap"] = True

        if gen <= 2:
            strategy["use_chunking"] = True
            strategy["notes"].append("LLM on low-gen chip → heavy chunking")

        elif gen >= 3:
            strategy["notes"].append("LLM on high-gen chip → optimized GGUF")

        # 🚨 EXTREME LOW RAM SAFETY
        # 🚨 CRITICAL RAM
    if ram_free < 1:
        strategy["precision"] = "int8"
        strategy["force_disk_offload"] = True
        strategy["use_chunking"] = True
        strategy["notes"].append("Critical RAM → INT8 + disk offload")

    # ⚠️ VERY LOW RAM
    elif ram_free < 1.5:
        strategy["force_disk_offload"] = True
        strategy["use_chunking"] = True
        strategy["notes"].append("Very low RAM → force disk-based execution")

# ⚠️ LOW RAM
    elif ram_free < 2:
        strategy["use_chunking"] = True
        strategy["notes"].append("Low RAM → avoid full model load")

    return strategy


if __name__ == "__main__":
    hw = detect_hardware()

    # Example test
    strategy = decide_optimization_strategy(hw, model_size_gb=6)

    from pprint import pprint
    print("=== HARDWARE ===")
    pprint(hw)

    print("\n=== STRATEGY ===")
    pprint(strategy)