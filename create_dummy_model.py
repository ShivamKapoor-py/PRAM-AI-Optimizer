import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(1000, 1000),
    nn.ReLU(),
    nn.Linear(1000, 1000)
)

torch.save(model, "real_model.pt")