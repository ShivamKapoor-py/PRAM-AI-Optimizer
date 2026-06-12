import torch


def split_model(model, num_chunks=2):
    layers = list(model.children())

    if len(layers) == 0:
        return [model]

    chunk_size = max(1, len(layers) // num_chunks)

    chunks = []
    for i in range(0, len(layers), chunk_size):
        sub = torch.nn.Sequential(*layers[i:i + chunk_size])
        chunks.append(sub)

    return chunks


def run_chunked_forward(chunks, x, device="cpu"):
    for chunk in chunks:
        chunk.to(device)
        x = chunk(x)
        chunk.to("cpu")  # free memory immediately
    return x