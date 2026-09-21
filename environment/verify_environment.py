"""Verify the development environment and record system information.

Run from the repository root with the virtual environment active:

    python environment/verify_environment.py

Checks that core packages import, PyTorch tensor ops and nn.Linear work,
MPS availability (CPU fallback), and that a saved model reloads with identical
output. Writes the results to environment/system_info.txt.
"""

from __future__ import annotations

import importlib
import platform
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

OUTPUT_PATH = Path(__file__).resolve().parent / "system_info.txt"
PACKAGES = ["numpy", "pandas", "matplotlib", "sklearn", "torch", "jupyterlab"]


def check(label: str, passed: bool, lines: list[str]) -> bool:
    status = "PASS" if passed else "FAIL"
    lines.append(f"[{status}] {label}")
    print(f"[{status}] {label}")
    return passed


def main() -> int:
    lines: list[str] = []
    ok = True

    lines.append("AI Independent Study 2026-2027 — Environment Record")
    lines.append(f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}")
    lines.append("")

    # ---- System ------------------------------------------------------------
    mac_ver = platform.mac_ver()[0]
    try:
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        chip = platform.processor() or "unknown"

    lines.append("== System ==")
    lines.append(f"OS: macOS {mac_ver} ({platform.platform()})")
    lines.append(f"Machine: {platform.machine()}")
    lines.append(f"CPU/Chip: {chip}")
    lines.append("")

    # ---- Python / venv ------------------------------------------------------
    in_venv = sys.prefix != sys.base_prefix
    lines.append("== Python ==")
    lines.append(f"Python version: {platform.python_version()}")
    lines.append(f"Python executable: {sys.executable}")
    lines.append(f"Virtual environment active: {in_venv}")
    lines.append("")

    # ---- Package imports ------------------------------------------------------
    lines.append("== Package versions ==")
    versions: dict[str, str] = {}
    for name in PACKAGES:
        try:
            module = importlib.import_module(name)
            versions[name] = getattr(module, "__version__", "unknown")
            lines.append(f"{name}: {versions[name]}")
        except Exception as exc:  # noqa: BLE001 - record any import failure
            lines.append(f"{name}: IMPORT FAILED ({exc!r})")
            ok = False
    lines.append("")

    # ---- PyTorch checks -------------------------------------------------------
    lines.append("== Checks ==")
    ok &= check("Python runs inside .venv", in_venv, lines)
    ok &= check("All packages import", len(versions) == len(PACKAGES), lines)

    import torch
    from torch import nn

    mps_available = torch.backends.mps.is_available()
    mps_built = torch.backends.mps.is_built()
    device = torch.device("mps" if mps_available else "cpu")

    x = torch.arange(6, dtype=torch.float32, device=device).reshape(2, 3)
    ok &= check(
        "Tensor operations",
        torch.allclose((x * 2 + 1).sum().cpu(), torch.tensor(36.0)),
        lines,
    )

    torch.manual_seed(0)
    layer = nn.Linear(3, 2).to(device)
    out = layer(x)
    ok &= check(
        f"nn.Linear forward pass (input {tuple(x.shape)} -> output {tuple(out.shape)})",
        out.shape == (2, 2) and torch.isfinite(out).all().item(),
        lines,
    )

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    model.eval()
    fixed_input = torch.linspace(-1.0, 1.0, steps=8).reshape(2, 4)
    with torch.no_grad():
        expected = model(fixed_input)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "tiny_model.pt"
        torch.save(model.state_dict(), path)
        reloaded = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
        reloaded.load_state_dict(torch.load(path, weights_only=True))
        reloaded.eval()
        with torch.no_grad():
            actual = reloaded(fixed_input)
    ok &= check("Model save/reload gives identical output", torch.equal(expected, actual), lines)
    lines.append("")

    lines.append("== PyTorch device ==")
    lines.append(f"torch version: {torch.__version__}")
    lines.append(f"MPS built: {mps_built}")
    lines.append(f"MPS available: {mps_available}")
    lines.append(f"Selected device: {device} (CPU is the fallback when MPS is unavailable)")
    lines.append(f"CUDA available: {torch.cuda.is_available()}")
    lines.append("")
    lines.append(f"Overall: {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")

    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nPython {platform.python_version()} | torch {torch.__version__} | "
          f"MPS available: {mps_available} | device: {device}")
    print(f"Recorded to {OUTPUT_PATH}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
