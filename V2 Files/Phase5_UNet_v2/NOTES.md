# Phase 5 (v2) — 3-class U-Net retrain

## Purpose

Implements the plan in `MODULE_V2_RETRAIN_JUSTIFICATION.md` in the project root. This folder is where you place **artefacts after** running:

- `module2_v2_multiclass_masks.ipynb`
- `module3_v2_unet_multiclass.ipynb`
- `module4_v2_stem_angle.ipynb`

## Files to copy here after a successful run (for slides / Drive zip)

| Source | What it is |
|--------|------------|
| `runs/unet_v2/best_unet_v2.pt` | v2 model weights (optional in zip if large) |
| `runs/unet_v2/unet_v2_confusion_matrix.png` | **3×3 pixel confusion (bg / fruit / ped)** — main rubric “confusion matrix” item |
| `runs/unet_v2/unet_v2_training_curves.png` | Train vs val loss and mIoU |
| `runs/unet_v2/unet_v2_predictions.png` | Qualitative 4-col / grid from notebook |
| `runs/unet_v2/confusion_val.csv` | Raw confusion counts |
| `runs/unet_v2/val_split_v2.json` | Which files were in val (reproducibility) |
| `runs/stem_angles_v3/stem_v3_examples.png` | PCA overlays from v2 peduncle channel |
| `runs/stem_angles_v3/stem_v3_dist.png` | Angle histogram after v2 |
| `runs/stem_angles_v3/stem_angles_v3.json` | Per-image angles |
| `multiclass_masks/sample_multiclass.png` | 3-class label sanity check (from module 2 v2) |

## Team: slide updates

1. **Replace or duplicate** the old Phase 5 U-Net slide: show **3-class** + confusion matrix; label axes **background / strawberry / peduncle** (not the sparse YOLO 1-class matrix).
2. **Comparison row:** v1 binary IoU (0.23 best) vs v2 **mean mIoU** and per-class — fill from notebook printout after you run.
3. **“Why we retrained”** one-liner: match peer evaluation + ROI + stratified split (see justification doc).
4. **Module 4 slide:** if you adopt v2 as final, reference **v3** outputs in `stem_angles_v3/` to avoid clashing with the earlier “v2 = threshold post-process on v1 U-Net” naming.

## Relation to v1 (Phase5_UNet_Training)

Keep the original `Phase5_UNet_Training/` folder as the **v1 baseline**. This `Phase5_UNet_v2/` folder is the **v2** story — both can appear in the report under “Model iterations.”
