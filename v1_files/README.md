# AIDA 2158A Final Project — Presentation Package
**Student:** Mark Miller
**Course:** Neural Networks and Deep Learning — Dr. M. Tufail, Red Deer Polytechnic
**Date:** April 2026

---

## How to Use This Package

This folder contains everything the team needs to build a presentation.

**Start here:**
1. Read `FINAL_REPORT.md` in this folder — it is the complete written report covering all phases with results, numbers, and analysis
2. Open each phase folder — each one has a `NOTES.md` with talking points and a description of every file inside it
3. Use the images and charts from each folder as slides

**Each phase folder is self-contained.** You do not need to open any notebooks or run any code. All outputs (charts, images, data) are already pre-generated and copied in.

---

## What Was Built

A complete deep learning pipeline for robotic strawberry harvesting:

```
Field photo → YOLOv11 detects strawberries → Crop around target fruit
           → U-Net segments the stem → PCA extracts stem angle
           → Gripper alignment angle for robot arm
```

---

## Folder Guide

### `FINAL_REPORT.md`
The full written project report. Covers all four modules, all results with numbers, honest analysis of limitations, and a rubric self-assessment. **Read this first.**

---

### `Phase1_Environment/`
**What:** Verified the Python/CUDA environment and installed all required packages.
**Key result:** NVIDIA RTX 5070 GPU confirmed, PyTorch + CUDA working, all packages installed.
| File | Use for |
|---|---|
| `NOTES.md` | Summary of environment setup, package versions, why `aida_stable` was chosen |
| `FINAL_REPORT.md` | Full report (same in every folder) |

---

### `Phase2_YOLOv11_Training/`
**What:** Trained YOLOv11 segmentation model on 2,800 strawberry images.
**Key result:** mAP50 = 0.927 (detection), 0.918 (segmentation masks) — excellent accuracy.

| File | Use for |
|---|---|
| `NOTES.md` | Talking points, hyperparameter table, results explained |
| `training_curves.png` | **Slide: show this** — loss and mAP over 48 epochs |
| `val_predictions_sample.png` | **Slide: show this** — 6 images with predicted masks overlaid |
| `confusion_matrix_normalized.png` | **Slide: show this** — how well the model classifies |
| `val_batch0_pred.jpg` / `val_batch1_pred.jpg` | Additional prediction examples |
| `BoxPR_curve.png` / `MaskPR_curve.png` | Precision-recall curves (box and mask) |
| `BoxP_curve.png` / `BoxR_curve.png` | Precision and recall curves individually (box) |
| `MaskP_curve.png` / `MaskR_curve.png` | Precision and recall curves individually (mask) |
| `results.png` | YOLO auto-generated results grid (all metrics per epoch) |
| `train_batch0.jpg` / `train_batch1.jpg` / `train_batch2.jpg` | Sample augmented training batches |
| `train_batch14000.jpg` / `train_batch14001.jpg` / `train_batch14002.jpg` | Late-epoch training batches |
| `val_batch2_labels.jpg` / `val_batch2_pred.jpg` | Additional val batch ground truth vs predictions |
| `results.csv` | Raw per-epoch numbers (open in Excel) |
| `module1_yolov11_training.ipynb` | Full code notebook |

---

### `Phase3_ROI_Crops/`
**What:** Used the trained YOLOv11 model to automatically crop a tight region around the largest strawberry in every image.
**Key result:** 3,100 / 3,100 images successfully cropped.

| File | Use for |
|---|---|
| `NOTES.md` | Talking points, method, confidence threshold decision explained |
| `roi_crops_sample.png` | **Slide: show this** — grid of sample crops |
| `module1_yolov11_training.ipynb` | Full code notebook (same as Phase 2 — ROI code is Cell 9) |

---

### `Phase4_Peduncle_Annotation/`
**What:** All four team members manually annotated the crown–stem–peduncle region using Roboflow + SAM. 357 annotated image/mask pairs produced.
**Key result:** 357 pairs total — 3.5× the 100-image minimum required by the spec.

| File | Use for |
|---|---|
| `NOTES.md` | Talking points, contributor breakdown, annotation method, format differences |
| `annotation_sample.png` | **Slide: show this** — 9 images with mask overlays |
| `contributor_breakdown.png` | **Slide: show this** — bar chart of annotations per contributor |
| `module2_peduncle_annotation.ipynb` | Full code notebook |

---

### `Phase5_UNet_Training/`
**What:** Trained a U-Net neural network to automatically segment the crown–stem–peduncle region.
**Key result:** Best val IoU = 0.2291 at epoch 10. Overfitting observed after epoch 10 — documented and explained.

| File | Use for |
|---|---|
| `NOTES.md` | Talking points, architecture, hyperparameters, honest assessment of results |
| `unet_training_curves.png` | **Slide: show this** — point out peak at epoch 10, then overfitting |
| `unet_predictions.png` | **Slide: show this** — 6 val images: original / ground truth / prediction / overlay |
| `module3_unet_training.ipynb` | Full code notebook |

---

### `Phase6_Stem_Angle/`
**What:** Applied PCA to U-Net predicted masks to extract the stem orientation angle — the robotic gripper alignment direction.
**Key result:** Mean stem angle 59.2° from horizontal across 50 val images. Before/after post-processing comparison included.

| File | Use for |
|---|---|
| `NOTES.md` | Talking points, what PCA is, v1 vs v2 comparison, angle interpretation |
| `v1_stem_angle_examples.png` | **Slide: show this first** — raw results showing noise problem |
| `v2_stem_angle_examples.png` | **Slide: show this second** — post-processed, cleaner axis lines |
| `v2_stem_angle_distribution.png` | **Slide: show this** — histogram + rose diagram of grasp directions |
| `v1_stem_angle_distribution.png` | v1 distribution for comparison |
| `v2_stem_angles.json` | Per-image angle data (all 57 val images) |
| `module4_stem_angle.ipynb` | Full code notebook |

---

## Suggested Presentation Slide Order

1. **Title slide** — project name, team, course
2. **Pipeline overview** — one diagram showing the 4-stage flow (in `FINAL_REPORT.md` Section 10)
3. **Phase 2: YOLOv11** — `training_curves.png` + `val_predictions_sample.png`
4. **Phase 3: ROI Crops** — `roi_crops_sample.png`
5. **Phase 4: Annotation** — `annotation_sample.png` + `contributor_breakdown.png`
6. **Phase 5: U-Net** — `unet_training_curves.png` + `unet_predictions.png`
7. **Phase 6: Stem Angle** — `v1_stem_angle_examples.png` → `v2_stem_angle_examples.png` → `v2_stem_angle_distribution.png`
8. **Results summary** — table from `FINAL_REPORT.md` Section 8 (rubric self-assessment)
9. **Limitations and future work** — `FINAL_REPORT.md` Section 9

---

## Key Numbers to Quote

| Metric | Value |
|---|---|
| YOLOv11 mAP50 (detection) | 0.927 |
| YOLOv11 mAP50 (segmentation) | 0.918 |
| ROI crops generated | 3,100 / 3,100 |
| Peduncle annotations | 357 pairs (4 contributors) |
| U-Net best val IoU | 0.2291 (epoch 10 / 50) |
| Stem angles extracted | 50 / 57 val images |
| Mean stem angle | 59.2° from horizontal |
