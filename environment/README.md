# Environment

Records of the development environment used for every experiment in this
repository.

| File | Purpose |
|---|---|
| `system_info.txt` | OS, chip, Python, package versions, MPS availability, and check results |
| `verify_environment.py` | Re-runs every check and rewrites `system_info.txt` |

## What the verification script checks

1. Python is running inside the `.venv` virtual environment.
2. `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `torch`, and `jupyterlab` import.
3. Basic tensor operations produce the expected result.
4. A tensor passes through `torch.nn.Linear` with the expected output shape.
5. `torch.backends.mps.is_available()`; the selected device falls back to CPU.
6. A small model saved with `torch.save` and reloaded gives identical output on a
   fixed input. The model is written to a temporary directory and deleted.

## Usage

From the repository root:

```bash
source .venv/bin/activate
python environment/verify_environment.py
```

The script exits with status `0` when all checks pass.

## When packages change

```bash
pip install <package>
python environment/verify_environment.py
pip freeze > requirements.txt
```

Commit `requirements.txt` and `system_info.txt` together so the recorded
environment matches the code that used it.
