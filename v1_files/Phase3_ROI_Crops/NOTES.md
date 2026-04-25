# Phase 3 — ROI Crop Generation
**AIDA 2158A Final Project | Mark Miller**

---

## What This Phase Did

Used the trained YOLOv11-seg model to automatically crop a tight region of interest (ROI) around the largest strawberry in every image. These crops are the input to Phase 4 (peduncle annotation) and Phase 5 (U-Net training).

---

## Why ROI Cropping?

The full images are 1008 × 756 pixels. A strawberry occupies only a small portion of each image. Cropping tightly around the target strawberry:

1. **Reduces the problem space** for the U-Net — it only needs to learn peduncle location within a local region, not search the entire image
2. **Removes distracting background** — other fruits, leaves, and stems from unrelated plants are excluded
3. **Matches the project pipeline** — the specification requires YOLOv11 to crop ROIs before peduncle annotation
4. **Improves U-Net training efficiency** — smaller input images = faster training, less memory

---

## How It Works

For each image:
1. Run YOLOv11 inference with `conf=0.01` (low threshold to catch all candidates)
2. Select the detection with the largest bounding box area — this is the "largest visible strawberry"
3. Add a 20-pixel pad on all sides for context (so the calyx/peduncle at the top isn't cut off)
4. Clamp to image boundaries
5. Save the crop as a PNG

**Why `conf=0.01`?** The model's confidence scores are naturally low on this dataset because each image contains many strawberries, so the model spreads confidence across many detections. The default YOLO threshold of 0.25 was filtering out most valid detections. Since we only need the *largest* detection regardless of confidence, using a very low threshold is correct here.

---

## Results

| Metric | Value |
|---|---|
| Total images processed | 3,100 |
| Successful crops saved | 3,100 (100%) |
| Images with no detection | 0 |
| Typical crop width | ~205 px |
| Typical crop height | ~223 px |
| Crop width range | 87 – 440 px |
| Crop height range | 46 – 433 px |
| Suspiciously tiny crops (<50px) | ~2 of 3,100 |

All 3,100 images were successfully cropped. The 2 tiny crops are edge cases where the detected strawberry was unusually small in the original photo — acceptable for a dataset of this size.

---

## Crop Selection Strategy: Largest by Bounding Box Area

We select the largest detection because:
- The project spec says "identify the largest visible ripe strawberry"
- Larger strawberries are more likely to be fully visible and ripe
- This gives a deterministic, reproducible selection criterion

A more sophisticated approach would also filter by colour (red vs green), but bounding box area alone gave clean results across the dataset.

---

## Output Location

All 3,100 crops are saved to:
```
Final Project\roi_crops\
```
Named identically to the source image (e.g. `217.png` → `roi_crops\217.png`).

**The originals are untouched** in their original train/val/test folders.

---

## Files in This Folder

| File | Description |
|---|---|
| `NOTES.md` | This file |
| `roi_crops_sample.png` | Grid of 9 sample ROI crops for presentation |
| `module1_yolov11_training.ipynb` | Notebook containing the crop code (Cell 9 and 10) |

---

## Presentation Talking Points

1. After training YOLOv11, we ran it on all 3,100 images to extract tight crops around the largest strawberry
2. 100% of images successfully cropped — no failures
3. Average crop size ~205 × 223 px, down from 1008 × 756 px — an 18× reduction in area
4. These crops are the foundation for peduncle annotation and U-Net training
5. Show: `roi_crops_sample.png`
