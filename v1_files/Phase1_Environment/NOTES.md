# Phase 1 — Environment Setup
**AIDA 2158A Final Project | Mark Miller**

---

## What This Phase Did

Verified and configured the Python environment so every subsequent phase can run without dependency issues.

---

## Environment Chosen: `aida_stable`

Out of three existing conda environments on this machine (`aida_stable`, `aida-gnn`, `aida2154_python_env`), **`aida_stable` was selected** because it was the only one that already had:
- PyTorch with a working CUDA build
- Ultralytics (YOLOv11) already installed
- JupyterLab and ipykernel

The other two environments were missing too many packages to be practical starting points.

---

## Packages Installed / Verified

| Package | Version | Role in project |
|---|---|---|
| Python | 3.11.14 | Runtime |
| PyTorch | 2.10.0+cu128 | Neural network training (YOLO, U-Net) |
| Ultralytics | 8.3.235 | YOLOv11-seg training and inference |
| NumPy | 2.2.6 | Array operations |
| Pandas | 3.0.2 | Results CSV parsing, hyperparameter tables |
| OpenCV | 4.12.0 | Image loading, contour extraction, ROI cropping |
| Pillow | 12.0.0 | Mask inspection |
| Matplotlib | 3.10.7 | Training curve plots |
| Seaborn | 0.13.2 | Available for visualisation |
| Scikit-learn | 1.8.0 | PCA (Phase 6 stem angle extraction) |
| GeoPandas | 1.1.3 | Installed per course requirements |
| PyArrow | 23.0.1 | Installed per course requirements |
| Statsmodels | 0.14.6 | Installed per course requirements |
| OpenPyXL | 3.1.5 | Excel output if needed |
| PyReadStat | 1.3.4 | Installed per course requirements |
| urllib3 | 2.6.0 | Installed per course requirements |
| ipykernel | 7.1.0 | Jupyter kernel integration |

**Packages added during this phase:** pandas, scikit-learn, seaborn, pyarrow, statsmodels, geopandas, openpyxl, pyreadstat

---

## GPU Confirmed

```
CUDA available: True
GPU: NVIDIA GeForce RTX 5070 Laptop GPU
```

This is critical — without CUDA, YOLOv11 training would take hours per epoch instead of minutes.

---

## Jupyter Kernel Registered

```
kernel name:    aida_stable
display name:   Python (aida_stable)
installed to:   C:\Users\markm\AppData\Roaming\jupyter\kernels\aida_stable
```

When opening any notebook in Jupyter, select **"Python (aida_stable)"** as the kernel.

---

## Why Not Create the `aida2158a` Environment from the Course Instructions?

The course setup instructions specify creating a fresh `aida2158a` environment with pinned older versions (e.g. NumPy 1.26.4, scikit-learn 1.4.1). We chose `aida_stable` instead because:

1. It already had PyTorch with CUDA — installing PyTorch from scratch is time-consuming and version-sensitive
2. It already had Ultralytics 8.3.235 (YOLOv11 support) — the course instructions predate YOLOv11
3. All required packages were installable into it without conflicts
4. Newer versions of packages are fully backwards-compatible for this project's needs

---

## Files in This Folder

| File | Description |
|---|---|
| `NOTES.md` | This file — phase summary and reasoning |

*No output artefacts — this phase was setup only.*
