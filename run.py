from optimizer.core import optimize
from optimizer.benchmark import benchmark

strategy = {
    #"device": "cpu",   # optional override
    "device": "mps",
    "use_chunking": True
}

model_path = "real_model.pt"

benchmark(model_path, optimize, strategy)