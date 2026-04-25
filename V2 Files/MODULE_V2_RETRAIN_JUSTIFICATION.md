# U-Net v2 — Retrain Justification and Run Order
**AIDA 2158A | Mark Miller (team use)**

## Why the v2 retrain exists

1. **Rubric and presentation feedback** — A single-class YOLOv11 model only produces a sparse 1-class confusion matrix. Other teams that trained a **3-class** segmentation model (background / fruit / peduncle) can show a full confusion matrix. The v2 U-Net is trained to output the same three classes so we can **report a comparable pixel-level confusion matrix** on validation data.

2. **Documented limitations of v1** — The v1 U-Net was binary, trained on **full 1008×756** frames where the peduncle was a tiny fraction of pixels, and the val split was **alphabetical** (effectively all one annotator). v2 is trained on **ROI crops** (same YOLO crop as Module 1) so fruit and peduncle occupy a larger share of the image, and the train/val split is **stratified by contributor** (random seed 42) so all four students appear in both splits.

3. **Training practice** — v2 uses **patience=5** early stopping (v1 used implicit patience=15 in YOLO only; the U-Net v1 kept training past the best val epoch). v2 also uses **CE + multi-class soft Dice** and **augmentation** (flip, small rotation, HSV jitter).

4. **Downstream (Module 4)** — A separate notebook (`module4_v2_stem_angle.ipynb`) takes **only predicted class 2 (peduncle)**, after `argmax`, and runs the same largest-component + PCA post-processing, saved under `runs/stem_angles_v3/`.

v1 (binary U-Net, full image, v1 stem angles) remains valid as a baseline and is **not deleted** — keep it in the report as “v1 / baseline” vs “v2 / improved design” if metrics improve; if v2 is still limited, explain honestly (data size, label noise).

---

## Run order (required)

1. **Module 1** must already be complete — trained YOLOv11 and weights at `runs/strawberry_seg/weights/best.pt` (or the alternate path listed in the notebook).
2. **`module2_v2_multiclass_masks.ipynb`**  
   - Reads `peduncle_masks/images/`, per-image YOLO labels from the four Roboflow exports, builds **3-class** label maps, crops with YOLO on each **full** image, writes:
   - `multiclass_masks/images/` (ROI RGB)
   - `multiclass_masks/masks_3class/` (pixel values 0, 1, 2)
3. **`module3_v2_unet_multiclass.ipynb`**  
   - Trains 3-class U-Net, writes `runs/unet_v2/best_unet_v2.pt`, `unet_v2_confusion_matrix.png`, `unet_v2_training_curves.png`, `unet_v2_predictions.png`, and **`val_split_v2.json`** (validation filenames — must be kept for step 4).
4. **`module4_v2_stem_angle.ipynb`**  
   - Loads v2 weights + `val_split_v2.json`, writes `runs/stem_angles_v3/`.

**Kernel:** `aida_stable` (or your env with PyTorch + Ultralytics + scikit-learn).

---

## What to copy into the presentation pack

After training, copy into `presentation/Phase5_UNet_v2/` (see that folder’s `NOTES.md`):

- From `runs/unet_v2/`: all PNG, CSV, JSON, and `best_unet_v2.pt` (optional for repo size — can omit weight from the zip and note “on request”).

---

## Slide / report updates for the team

- **Method slide:** “Binary U-Net on full image (v1)” → “3-class U-Net on YOLO ROI crops (v2)”.
- **Results slide:** Add **v2 mIoU** and per-class IoU; show **`unet_v2_confusion_matrix.png`**; keep v1 numbers in a small “baseline” row if you want history.
- **Module 4:** If using v2 pipeline for the final story, cite **v3** stem angle outputs and `stem_angles_v3.json` (mean angle will differ from v2 post-process on v1 U-Net).
- **Summary / limitations:** State that aida2154’s peduncle was originally bbox labels — v2 still uses those rectangles for class 2 where applicable.

Do **not** claim “optimal” or “real-world deployment” — v2 is a stronger **experimental** design; metrics still depend on only ~357 annotated images.

---

## No changes required to Module 1

YOLOv11 training and `roi_crops/` generation are unchanged. v2 reuses the same YOLO `best.pt` for crop geometry consistency with Module 1.
