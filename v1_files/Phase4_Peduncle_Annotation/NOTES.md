# Phase 4 — Peduncle Annotation: Merging All Contributors
**AIDA 2158A Final Project | Mark Miller**

---

## What This Phase Did

Collected manual peduncle annotations from all four contributors, converted them from YOLO polygon format into binary PNG masks, and merged everything into one unified dataset ready for U-Net training.

---

## What Was Annotated

The crown–stem–peduncle region of the target strawberry in each image, including:
- The green stem visible above the fruit
- The calyx (leafy crown attached to the top of the fruit)
- A small upper portion of the strawberry body for biological continuity

**Not annotated:** hidden stems, unrelated fruits, background leaves.

This follows the project specification exactly (Section 6 — Important Annotation Rule).

---

## Contributors

| Contributor | Tool | Images | Format received | Peduncle class |
|---|---|---|---|---|
| hervejunior | Roboflow | 100 | YOLO polygon segmentation | Class 0 |
| biberdork | Roboflow | 100 | YOLO polygon segmentation | Class 0 |
| aida2154 | Roboflow | 90 | YOLO **bounding box** (not polygon) | Class 0 |
| markm (Mark Miller) | Roboflow + SAM3 | 96 | YOLO polygon segmentation | Class 0 |

**Note on aida2154:** Their dataset was exported in Object Detection format (bounding boxes) rather than Instance Segmentation format (polygons). Their peduncle bounding boxes were converted to filled rectangular masks. This is a fair representation of their annotation intent.

---

## Annotation Method (Mark Miller)

1. Uploaded 96 images from the shared dataset to `app.roboflow.com`
2. Used **SAM3 Smart Polygon** mode — hover over peduncle region, SAM auto-generates mask
3. Refined each mask manually if needed
4. Labelled as class `Peduncle`
5. Exported as YOLOv11 segmentation format

This is equivalent to the SAM-assisted Digital Sreeni workflow described in the project spec — Roboflow's SAM3 implementation performs the same function.

---

## Conversion Process

For each contributor's dataset:
1. Read each image + matching `.txt` label file
2. Extract only annotations where class == Peduncle (class 0)
3. For **polygon** labels: use `cv2.fillPoly()` to render white filled region on black canvas
4. For **bounding box** labels (aida2154): use `cv2.rectangle()` to fill the bounding box region
5. Save image → `peduncle_masks/images/CONTRIBUTOR_stem.png`
6. Save mask → `peduncle_masks/masks/CONTRIBUTOR_stem.png`

---

## Final Dataset

| Contributor | Masks produced |
|---|---|
| hervejunior | 98 |
| biberdork | 100 |
| aida2154 | 84 |
| markm | 75 |
| **Total** | **357** |

- All 357 pairs verified: images match masks, no empty masks
- All source images are 1008×756 pixels
- Masks are binary: 255 = peduncle region, 0 = background

**The project spec required 100 annotations. This team produced 357 — 3.5× the minimum.**

---

## Why Some Images Were Skipped

- hervejunior/biberdork: 2 images had no peduncle annotation (strawberry body only)
- aida2154: 6 images had no class 0 annotation (strawberry body only)
- markm: 21 images were duplicated by Roboflow augmentation and deduplicated

---

## Files in This Folder

| File | Description |
|---|---|
| `NOTES.md` | This file |
| `module2_peduncle_annotation.ipynb` | Full conversion notebook with all code |
| `annotation_sample.png` | Grid of 9 image+mask overlay samples |
| `contributor_breakdown.png` | Bar chart of annotation counts by contributor |

---

## Presentation Talking Points

1. All four team members contributed manual peduncle annotations using Roboflow
2. Mark used SAM3 (Segment Anything Model) for assisted annotation — matches project spec intent
3. Total: **357 annotated pairs** — 3.5× the 100-image minimum required
4. One dataset (aida2154) used bounding box format instead of polygons — handled in code
5. All annotations converted to unified binary masks (white = peduncle, black = background)
6. Show: `annotation_sample.png`, `contributor_breakdown.png`
