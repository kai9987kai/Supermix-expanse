import os
from typing import Any, Dict, Tuple

import torch


def configure_torch_runtime(
    torch_num_threads: int = 0,
    torch_interop_threads: int = 0,
    allow_tf32: bool = True,
    matmul_precision: str = "high",
    strict_determinism: bool = False,
) -> None:
    cpu_count = os.cpu_count() or 1
    n_threads = int(torch_num_threads)
    if n_threads <= 0:
        n_threads = cpu_count
    try:
        torch.set_num_threads(max(1, n_threads))
    except Exception:
        pass

    interop = int(torch_interop_threads)
    if interop <= 0:
        interop = min(8, max(1, cpu_count // 2))
    try:
        torch.set_num_interop_threads(max(1, interop))
    except Exception:
        pass

    try:
        torch.set_float32_matmul_precision(str(matmul_precision))
    except Exception:
        pass

    if hasattr(torch.backends, "cuda") and torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
            torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
            if bool(strict_determinism):
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                torch.use_deterministic_algorithms(True, warn_only=True)
            else:
                torch.backends.cudnn.benchmark = True
        except Exception:
            pass


def _try_directml() -> Any:
    try:
        import torch_directml  # type: ignore

        return torch_directml.device()
    except Exception:
        return None


def resolve_device(device_spec: str = "auto", preference: str = "cuda,npu,xpu,dml,mps,cpu") -> Tuple[Any, Dict[str, str]]:
    spec = str(device_spec or "auto").strip().lower()
    if spec and spec != "auto":
        if spec == "dml":
            dml = _try_directml()
            if dml is None:
                raise RuntimeError("Requested device 'dml' but torch_directml is not available.")
            return dml, {"requested": spec, "resolved": "dml"}
        return torch.device(spec), {"requested": spec, "resolved": spec}

    order = [x.strip().lower() for x in str(preference or "").split(",") if x.strip()]
    if not order:
        order = ["cuda", "npu", "xpu", "dml", "mps", "cpu"]

    for kind in order:
        if kind == "cuda":
            try:
                if torch.cuda.is_available():
                    return torch.device("cuda"), {"requested": "auto", "resolved": "cuda"}
            except Exception:
                pass
        elif kind == "xpu":
            try:
                if hasattr(torch, "xpu") and torch.xpu.is_available():  # type: ignore[attr-defined]
                    return torch.device("xpu"), {"requested": "auto", "resolved": "xpu"}
            except Exception:
                pass
        elif kind == "mps":
            try:
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return torch.device("mps"), {"requested": "auto", "resolved": "mps"}
            except Exception:
                pass
        elif kind == "npu":
            try:
                if hasattr(torch, "npu") and torch.npu.is_available():  # type: ignore[attr-defined]
                    return torch.device("npu"), {"requested": "auto", "resolved": "npu"}
            except Exception:
                pass
            try:
                import torch_npu  # type: ignore # noqa: F401

                if hasattr(torch, "npu") and torch.npu.is_available():  # type: ignore[attr-defined]
                    return torch.device("npu"), {"requested": "auto", "resolved": "npu"}
            except Exception:
                pass
        elif kind == "dml":
            dml = _try_directml()
            if dml is not None:
                return dml, {"requested": "auto", "resolved": "dml"}
        elif kind == "cpu":
            return torch.device("cpu"), {"requested": "auto", "resolved": "cpu"}

    return torch.device("cpu"), {"requested": "auto", "resolved": "cpu"}
