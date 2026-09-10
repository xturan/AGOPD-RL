from __future__ import annotations

import importlib
import sys


def main() -> None:
    print(f"python={sys.executable}")
    check_torch()
    for module_name in [
        "numpy",
        "regex",
        "transformers",
        "huggingface_hub",
        "datasets",
        "ray",
        "tensordict",
        "torchdata",
        "verl",
    ]:
        check_module(module_name)
    check_module("vllm", optional=True)


def check_torch() -> None:
    try:
        import torch

        print(f"torch={torch.__version__}")
        print(f"torch_cuda={torch.version.cuda}")
        print(f"cuda_available={torch.cuda.is_available()}")
        print(f"gpu_count={torch.cuda.device_count()}")
        if torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                print(f"gpu_{idx}={torch.cuda.get_device_name(idx)}")
    except Exception as exc:
        print(f"torch=FAILED {exc!r}")


def check_module(module_name: str, *, optional: bool = False) -> None:
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "unknown")
        file = getattr(module, "__file__", None)
        print(f"{module_name}={version} file={file}")
    except Exception as exc:
        status = "OPTIONAL_MISSING" if optional else "FAILED"
        print(f"{module_name}={status} {exc!r}")


if __name__ == "__main__":
    main()
