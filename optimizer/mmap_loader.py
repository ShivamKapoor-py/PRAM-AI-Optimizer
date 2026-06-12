import mmap


def load_mmap(file_path):
    """
    Memory-map GGUF file (real RAM reduction)
    """
    f = open(file_path, "rb")

    mm = mmap.mmap(
        f.fileno(),
        length=0,
        access=mmap.ACCESS_READ
    )

    return mm