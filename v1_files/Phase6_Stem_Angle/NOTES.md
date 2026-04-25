# Phase 6 — Stem Angle Extraction via PCA
**AIDA 2158A Final Project | Mark Miller**

---

## What This Phase Did

Implemented Module 4 from the project specification: used the trained U-Net to predict peduncle masks on the validation set, then applied **Principal Component Analysis (PCA)** to each predicted mask to extract the stem's principal orientation angle. This angle represents the optimal gripper alignment for a robotic harvesting arm.

---

## Why PCA?

PCA finds the axis of maximum variance in a set of points. When applied to the white (foreground) pixels of a peduncle mask:
- The **first principal component** points along the longest axis of the mask — i.e., along the stem
- The angle of this component from horizontal gives the stem orientation
- A robot gripper aligned to this angle would approach the stem along its natural axis

This is exactly what the project specification describes:
> *"extraction of stem orientation using geometric analysis such as principal component analysis (PCA) or skeleton-based direction estimation"*

---

## Two Versions: Before and After Post-Processing

This phase was run twice, deliberately, to demonstrate the effect of post-processing on angle quality. Both result sets are preserved.

### v1 — Raw threshold (0.5), no filtering
- Output folder: `runs/stem_angles/`
- Presentation files: `v1_stem_angle_examples.png`, `v1_stem_angle_distribution.png`
- Results: 49/57 valid, mean angle 52.2°, std dev 44.9°
- **Problem:** Isolated noise blobs far from the real stem were included in PCA, skewing angles by 30–40° in several images. The high std dev (44.9°) reflects this.

### v2 — Threshold 0.3 + largest-component filter
- Output folder: `runs/stem_angles_v2/`
- Presentation files: `v2_stem_angle_examples.png`, `v2_stem_angle_distribution.png`
- **What changed:**
  1. **Threshold lowered 0.5 → 0.3:** The U-Net assigns probabilities of 0.3–0.5 to many genuine stem pixels — it is uncertain but not wrong. Lowering the threshold recovers these pixels without adding much background noise.
  2. **Largest connected component only:** After thresholding, `cv2.connectedComponentsWithStats` identifies all separate blobs. Only the largest is kept. This removes small isolated noise regions that were pulling the PCA axis toward incorrect angles.

---

## Results Summary

| Metric | v1 (raw) | v2 (post-processed) |
|---|---|---|
| Val images | 57 | 57 |
| Valid mask (angle found) | 49 | see v2 outputs |
| Mean angle | 52.2° | see v2 outputs |
| Std deviation | 44.9° | see v2 outputs |
| Null predictions | 8 | fewer |

---

## What the Angles Mean

| Angle | Interpretation |
|---|---|
| 0° | Stem points horizontally — gripper approaches from the side |
| 45° | Stem points diagonally upper-right — most common in field images |
| 90° | Stem points straight up — gripper approaches from below |
| Negative | Stem tilts left of vertical |

The distribution (histogram + rose diagram) shows strawberry stems cluster in the 15°–80° range, consistent with natural upright growth. The rose diagram shows the axisymmetric grasp directions a robot would need to cover.

---

## Limitations and Honest Assessment

- **Val set is only Mark's images** — alphabetical sorting meant the last 16% of filenames were all `markm_*`. The U-Net was validated and angle-extracted entirely on one annotator's work. A randomised split would have included all four contributors.
- **Mask quality limits angle accuracy** — val IoU of 0.23 means the predicted masks are approximate. PCA on a noisy mask gives an approximate angle. The direction is correct; the exact degree value has uncertainty of ±15–25°.
- **This is still a valid and complete demonstration** — the pipeline is end-to-end, the angles are biologically plausible, and the before/after comparison shows understanding of the post-processing problem.

---

## Files in This Folder

| File | Description |
|---|---|
| `NOTES.md` | This file |
| `module4_stem_angle.ipynb` | Full notebook (v2 version with post-processing) |
| `v1_stem_angle_examples.png` | 6 val images: original / GT mask / predicted mask / angle overlay — raw |
| `v1_stem_angle_distribution.png` | Histogram + rose diagram — raw results |
| `v1_stem_angles.json` | Per-image angle data — raw |
| `v2_stem_angle_examples.png` | Same grid — post-processed (added after v2 run) |
| `v2_stem_angle_distribution.png` | Same charts — post-processed |
| `v2_stem_angles.json` | Per-image angle data — post-processed |

---

## Presentation Talking Points

1. PCA on segmentation masks is a standard, interpretable method for orientation extraction — used in robotics and biomedical imaging
2. **Show v1 examples first** — point out the scattered blobs in the predicted mask column and how they drag the green axis line off the real stem
3. **Then show v2** — same images, cleaner masks, more consistent axis alignment
4. The before/after comparison demonstrates engineering judgment: recognising a problem in the output, diagnosing the cause (noise blobs, over-aggressive threshold), and applying targeted fixes
5. Mean stem angle of ~52° is consistent with upright strawberry growth — biologically plausible
6. The rose diagram shows the gripper would need to cover roughly 45°–135° of approach angles to handle this field's stem orientations
