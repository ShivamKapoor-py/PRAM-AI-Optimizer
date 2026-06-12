import os
import sys
import time
import torch
import subprocess
import logging

# 1. SETUP LOGGING (Industry Standard)
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("PRAM-Engine")

# 2. ENVIRONMENT & PATH ALIGNMENT
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

try:
    from model.analyzer import analyze_model
    from hardware.detect import detect_hardware
    from optimizer.core import load_model, optimize_mps, optimize_cpu
    from optimizer.gguf import convert_to_gguf
    from optimizer.benchmark import benchmark_gguf, get_file_size_gb
    from optimizer.disk_splitter import split_model
except ImportError as e:
    logger.error(f"❌ Critical: Project Structure Error -> {e}")
    sys.exit(1)

# REFERENCE PATHS
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

HF_MODEL_DEFAULT = os.path.join(MODELS_DIR, "hf_model")
REAL_MODEL_DEFAULT = os.path.join(BASE_DIR, "real_model.pt")
F16_GGUF = os.path.join(MODELS_DIR, "model-f16.gguf")
Q4_GGUF = os.path.join(MODELS_DIR, "model-q4.gguf")

def main(target_path=None):
    """
    Main Optimization Entry Point.
    :param target_path: Optional path passed from SDK/CLI.
    """
    print("\n" + "═"*60)
    print("🚀 PRAM: HARDWARE-AWARE AI OPTIMIZATION ENGINE")
    print("═"*60)

    # --- STEP 1: HARDWARE DETECTION ---
    hw = detect_hardware()
    # Industry standard check for Apple Silicon vs Intel/Generic
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"📍 Hardware: {hw.get('chip_info', {}).get('chip_name', 'Generic')} | RAM: {hw.get('ram_total_gb')}GB")

    # --- STEP 2: DYNAMIC PATH RESOLUTION ---
    if target_path is None:
        target_path = HF_MODEL_DEFAULT if os.path.exists(HF_MODEL_DEFAULT) else REAL_MODEL_DEFAULT
    
    if not os.path.exists(target_path):
        logger.error(f"❌ Error: Model path not found: {target_path}")
        return

    # Analyze metadata
    analysis = analyze_model(target_path)
    size_gb = analysis.get('model_size_gb', 0)
    is_directory = os.path.isdir(target_path)
    
    logger.info(f"[STEP 1] Model: {os.path.basename(target_path)} ({size_gb} GB)")

    start_time = time.time()
    final_path = ""

    # --- STEP 3: THE JUDGE (ROUTING LOGIC) ---
    try:
        # FOLDERS (HF) or LARGE MODELS (>4GB) go to GGUF
        if is_directory or size_gb > 4.0:
            logger.info("➡️ Strategy: GGUF Quantization Pipeline (High Compression)")
            
            # Step 3a: F16 Conversion
            if not os.path.exists(F16_GGUF):
                logger.info("...Converting to GGUF F16 format")
                convert_to_gguf(target_path, F16_GGUF, quant="f16")
            
            # Step 3b: Q4 Quantization
            quant_bin = os.path.join(BASE_DIR, "llama.cpp/build/bin/llama-quantize")
            if not os.path.exists(quant_bin):
                logger.error(f"❌ Missing llama-quantize binary at {quant_bin}")
                return

            logger.info("...Executing 4-bit integer quantization (Q4_0)")
            subprocess.run([quant_bin, F16_GGUF, Q4_GGUF, "q4_0"], check=True, capture_output=True)
            final_path = Q4_GGUF

        # SINGLE FILE SMALL MODELS (<=4GB) go to PyTorch Optimization
        else:
            logger.info("➡️ Strategy: Native PyTorch Pipeline (Memory Streaming)")
            model = load_model(target_path)
            
            if device == "mps":
                logger.info("...Applying FP16 Metal Shaders + Disk Splitting")
                model = optimize_mps(model)
                split_model(model, num_chunks=2)
                final_path = target_path.replace(".pt", "_opt_fp16.pt")
            else:
                logger.info("...Applying INT8 Dynamic Quantization")
                model = optimize_cpu(model)
                final_path = target_path.replace(".pt", "_opt_int8.pt")
            
            torch.save(model, final_path)

    except Exception as e:
        logger.error(f"❌ Optimization Failed: {str(e)}")
        return

    # --- STEP 4: ANALYTICS & REPORTING ---
    end_time = time.time()
    size_after = get_file_size_gb(final_path)
    reduction_pct = ((size_gb - size_after) / size_gb) * 100 if size_gb > 0 else 0

    print("\n" + "═"*60)
    print("💎 OPTIMIZATION REPORT")
    print("═"*60)
    print(f"{'Source Size':<25} : {size_gb:>8} GB")
    print(f"{'Optimized Size':<25} : {round(size_after, 2):>8} GB")
    print(f"{'Memory Reduction':<25} : {round(reduction_pct, 1):>8}%")
    print(f"{'Process Time':<25} : {round(end_time - start_time, 2):>8}s")
    print("═"*60 + "\n")
    
    # Run performance benchmark for GGUF models
    if final_path.endswith(".gguf") and os.path.exists(F16_GGUF):
        benchmark_gguf(F16_GGUF, Q4_GGUF, device)

if __name__ == "__main__":
    # Allow CLI usage: python run_optimizer.py path/to/model.pt
    input_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(input_path)