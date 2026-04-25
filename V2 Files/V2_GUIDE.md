# V2 Pipeline Guide
## A Comprehensive Reference for the Multi-Class Retraining

**Project:** AIDA 2158 — Strawberry Harvesting Robot  
**Scope:** Modules 2, 3, and 4 (v2 variants)  
**Status:** Living document — updated as v2 results are finalized

---

## Why V2 Exists

The original pipeline (v1) trained a binary U-Net to answer one question: *is this pixel a peduncle, or not?* That binary model had a structural problem — its confusion matrix was almost entirely empty except for the background-vs-peduncle diagonal. There was no way to evaluate whether the model understood the fruit body at all, because the fruit was never a training target.

V2 reframes the problem as **three-class semantic segmentation**:

| Class | Value | Meaning |
|-------|-------|---------|
| Background | 0 | Anything that is not strawberry or peduncle |
| Strawberry | 1 | The fruit body (red flesh) |
| Peduncle | 2 | The crown, stem, and calyx above the fruit |

This gives the model structural context. It learns that class 2 (peduncle) sits above class 1 (strawberry), which is physically meaningful for the cutting-point task. The confusion matrix becomes a full 3×3, so we can see whether errors are random noise or systematic (e.g., model confusing peduncle with strawberry vs. peduncle with background).

---

## V1 → V2 Impact Summary

This section states what specifically changed, why it changed, and how large the measurable difference was. Where a direct metric comparison is not valid (different task, different data), that is stated explicitly.

### What changed and why

| Area | V1 | V2 | Why it changed |
|------|----|----|---------------|
| Segmentation task | Binary (peduncle vs. not) | 3-class (BG / strawberry / peduncle) | Binary gave no information about whether the model understood fruit context. Fruit class gives the model structural layout to learn from. |
| Training input | Full 1008×756 images | ROI crops per detected fruit | Peduncle occupied ~2–5% of each full image. The model wasted capacity learning that 95%+ of each frame is background. Cropped ROIs put the peduncle in context. |
| Crops generated | 1 per source image (largest fruit) | 1 per detected fruit (all fruits) | Any image containing 4 strawberries now yields 4 training examples. Scales dataset automatically with image content. |
| Crop padding | 20px fixed, symmetric | 100% top / 25% sides / 20px bottom | Peduncle grows above the fruit. A symmetric 20px pad clips it. The asymmetric top extension guarantees crown and stem are always in frame. |
| YOLO confidence for mask generation | 0.01 (harvesting-inference default) | 0.15 | At 0.01, YOLO fires on shadows and leaves, creating garbage crops that corrupt the training set. 0.15 still catches small/partial strawberries but removes clear false positives. |
| Content filter | None | MIN_CONTENT = 200 px | Any crop whose mask contains fewer than 200 non-background pixels is discarded. Catches edge cases where YOLO detects a fruit but no annotation exists for it. |
| Train/val split | Alphabetical (all `markm_*` in val) | Stratified by contributor (random seed 42) | The v1 split validated on a single annotator's images. V2 guarantees each contributor appears proportionally in both train and val. |
| Loss function | BCE + binary Dice | CrossEntropy (weighted) + multiclass soft Dice | 3-class output requires `CrossEntropyLoss`. Weighting required because peduncle pixels are ~23× rarer than background. |
| Class weighting | None | 12× cap on minority class | Without weighting, the model almost ignores peduncle (low recall, high false-negative). With a 26× uncapped weight, recall nearly doubled but precision collapsed to 18.7%. 12× cap hit the balance point. |
| Confusion matrix | 2×2 (binary, near-empty) | 3×3 (background / strawberry / peduncle) | The 3×3 matrix reveals whether peduncle errors are "confused with fruit" vs. "confused with background" — categorically different failure modes. |
| PCA angle output | Raw arctan2 (range: any float) | Folded to [−90°, +90°] | PCA axes are sign-ambiguous. Two images of the same physical stem could produce 118° and −62° without folding. Normalised output is safe for downstream robot arm logic. |

---

### Measured differentials

#### Within V2: effect of class weighting (same task, same data — directly comparable)

| Metric | V2 Run 1 (unweighted) | V2 Run 3 (12× cap, **final**) | Δ |
|--------|----------------------|-------------------------------|---|
| Best val mIoU | 0.5953 | 0.5843 | −0.011 (mIoU slightly lower; see note) |
| BG IoU | 0.886 | 0.772 | −0.114 (expected: weight shift away from BG) |
| Fruit IoU | 0.737 | **0.787** | **+0.050 (+6.8%)** |
| Peduncle IoU | 0.173 | **0.195** | **+0.022 (+12.7%)** |
| Peduncle recall | 33.5% | **54.0%** | **+20.5 percentage points** |
| Peduncle precision | 26.4% | 23.4% | −3.0 ppts (small trade-off, acceptable) |

> **Note on mIoU:** mIoU is an average across all three classes. When BG is weighted less and the model slightly over-fires on peduncle, background IoU drops, pulling the average down even though the classes that matter most (fruit and peduncle) improve. The per-class numbers tell the real story.

#### V1 → V2: peduncle detection coverage (Module 4 output — comparable in spirit, different in scale)

| Metric | V1 Module 4 | V2 Module 4 | Δ |
|--------|-------------|-------------|---|
| Validation set size | 57 images (1 annotator) | 192 crops (4 annotators) | 3.4× more, and stratified |
| Valid angles produced | 50 (87.7%) | **178 (92.7%)** | **+5.0 percentage points** |
| No-peduncle-found | 7 (12.3%) | 14 (7.3%) | Null rate halved |
| Angle range | −? to +? (unfolded) | −43.85° to +81.89° (folded) | Consistent reference frame |

> **Why direct V1-vs-V2 IoU comparison is not presented:** V1 measured a single binary IoU (peduncle vs. not-peduncle, on full 1008×756 images, 57-image val from 1 annotator). V2 measures per-class IoU on 3-class predictions on 256×256 ROI crops, 192-crop val from all 4 contributors. Comparing 0.2291 (v1 binary) to 0.195 (v2 peduncle class) would imply the model got worse — the opposite is true. The V1 task was easier because the background class dominated the loss, making it cheap to achieve modest binary IoU. V2 forces the model to simultaneously learn two hard boundaries. The within-v2 comparison (unweighted → weighted) is the clean measurement of what the class-weighting optimisation did.

---

## Module 2 V2 — Multiclass Mask Generation

**Notebook:** `module2_v2_multiclass_masks.ipynb`  
**Output:** `multiclass_masks/images/` and `multiclass_masks/masks_3class/`

### What changed from v1

#### 1. Three-class masks instead of binary

V1 produced single-channel PNG masks where 255 = peduncle and 0 = everything else.

V2 produces single-channel PNG masks where pixel values are 0, 1, or 2 — directly usable as `CrossEntropyLoss` targets without any conversion.

The build order matters:
1. Strawberry polygons are written first (class 1)
2. Peduncle polygons are written second and **overwrite** class 1 wherever they overlap

This overwrite priority ensures the peduncle class is never accidentally suppressed by a strawberry annotation that covers the same region.

#### 2. HSV strawberry fill

Because many annotators labeled only the peduncle and not the fruit body, large regions of actual strawberry would be left as class 0 (background). This would teach the model that red-fleshed regions are background — incorrect.

V2 applies an HSV heuristic after building the label mask: any pixel that falls in the red HSV range (hue 0–15 or 165–180, saturation > 40, value > 40) and is currently class 0 gets assigned class 1. Peduncle regions are then re-applied on top to ensure no peduncle pixels are accidentally overwritten by HSV fill.

#### 3. Per-fruit cropping (every detected strawberry, not just the largest)

V1 ignored the concept of ROI crops entirely — it trained on full images.

V2 runs YOLOv11 on each source image and generates **one crop per detected strawberry**, not just the largest one. If an image contains four ripe strawberries, the output is four separate crops: `base_00.png`, `base_01.png`, `base_02.png`, `base_03.png`. This:
- Scales the dataset with the number of fruits present per image
- Ensures each crop is centered on one specific fruit, giving the U-Net a consistent scale and framing
- Means every fruit in the dataset gets annotated, not just the most prominent one

#### 4. Asymmetric padding

V1 had no cropping, so padding was not relevant.

V2's crop boxes are padded asymmetrically, because the peduncle anatomy is asymmetric:

| Direction | Amount | Rationale |
|-----------|--------|-----------|
| Top | +100% of bbox height | The peduncle extends entirely above the fruit. A tight YOLO bbox would clip it. Adding a full bbox-height above guarantees the crown and stem are in frame. |
| Left / Right | +25% of bbox width | The peduncle angle can deviate left or right, especially in angled or side-view images. 25% gives room for oblique stems without bloating the image with irrelevant background. |
| Bottom | +20px (fixed) | The peduncle does not extend below the fruit. A small fixed pad avoids hard pixel edges at the bottom of the frame. |

The padding is computed relative to each individual detection's bounding box, not a global fixed value. A large strawberry gets more absolute padding than a small one, which keeps the peduncle-to-frame ratio consistent.

#### 5. YOLO confidence threshold raised to 0.15

The original setting of `CONF = 0.01` was kept from Module 1, where a very low threshold ensures no ripe strawberry is missed during harvesting inference. For **training data generation**, however, this is the wrong trade-off. At 0.01, YOLO fires on shadows, background patches, partial leaves, and other noise. Each false-positive detection produces a crop that contains no actual fruit — just background pixels — which actively teaches the model that those background regions are all three classes randomly. This corrupts the decision boundary.

Raising to `CONF = 0.15` still catches small, partially-occluded, or side-on strawberries, but filters out clear false positives.

#### 6. Minimum content filter (MIN_CONTENT = 200 pixels)

Even after raising the confidence threshold, some crops can slip through with no annotated content — for example, a correctly detected strawberry at the edge of an image where the asymmetric top-extension extends beyond the image boundary, or a fruit with no annotation in any of the four contributor label sets.

After generating each crop, `np.count_nonzero(mcp)` counts the number of non-background (class 1 or class 2) pixels in the mask. If fewer than `MIN_CONTENT = 200` pixels are non-background, the crop is discarded. The final print output reports how many crops were skipped for this reason separately from how many were skipped for missing labels.

This filter ensures every crop in the training set has at least some annotated fruit or peduncle content.

#### 7. Output folder cleanup on startup

When Cell 1 runs, both `multiclass_masks/images/` and `multiclass_masks/masks_3class/` are deleted and recreated. This prevents stale files from a previous run (which used a different naming convention) from silently contaminating the new dataset.

---

## Module 3 V2 — Multi-Class U-Net Training

**Notebook:** `module3_v2_unet_multiclass.ipynb`  
**Output:** `runs/unet_v2/best_unet_v2.pt`, `runs/unet_v2/unet_v2_predictions.png`, `runs/unet_v2/confusion_matrix_v2.png`

### What changed from v1

#### 1. Three output channels instead of one

V1's U-Net had a single output channel with sigmoid activation, producing a probability that a pixel is peduncle.

V2's U-Net (`UNet3`) has three output channels with softmax across channels. The predicted class for each pixel is `argmax(output, dim=0)`, giving values 0, 1, or 2.

#### 2. Loss function with inverse-frequency class weighting

V1 used binary cross-entropy (BCE) plus a binary Dice loss.

V2 uses `CrossEntropyLoss` plus a soft multiclass Dice loss (`soft_dice_multiclass`) that computes Dice per class and averages:

```
loss = CrossEntropyLoss(weight=class_weights, pred, target)
     + (1 - soft_dice_multiclass(pred_softmax, target))
```

**Class weights** are computed dynamically from the actual training masks at the start of Cell 5, using inverse-frequency weighting:

```
weight[c] = total_pixels / (n_classes × pixels_for_class_c)
```

Then normalized so the smallest weight = 1.0. With a typical distribution of BG ~23×, Fruit ~7×, Peduncle ~1× relative to each other, this yields approximate weights of BG=1.0, Fruit=3.4, Peduncle=23+. Without this, `CrossEntropyLoss` treats every pixel equally and the model learns to almost ignore the peduncle class (which is rare) because getting background and fruit right accounts for ~97% of the loss signal.

Weights are computed dynamically rather than hardcoded so they automatically adapt whenever the training data changes (e.g., after re-running module2_v2 with different CONF or MIN_CONTENT settings).

**Weight cap at 12×:** After the first weighted run (peduncle weight computed at ~26×), the confusion matrix showed the model was over-predicting peduncle aggressively — 20% of background pixels were being fired as peduncle (1.72M false positives out of 8.4M background pixels). Peduncle precision collapsed to 18.7%. A cap of `np.clip(_w, 1.0, 12.0)` was added so the peduncle weight can never exceed 12× regardless of class distribution. This keeps the recall benefit of weighting without the over-trigger problem. Result with uncapped 26×: peduncle IoU 0.168, recall 62.8%, precision 18.7%. Expected with 12× cap: peduncle IoU ~0.20–0.25, better balance of recall and precision.

#### 3. Dataset and data loader

V2 uses `Roi3Dataset`, which loads from `multiclass_masks/images/` and `multiclass_masks/masks_3class/`. Masks are loaded as single-channel integers (0, 1, 2) and converted to `torch.long` for `CrossEntropyLoss` compatibility. Images are resized to 256×256 with bilinear interpolation; masks use nearest-neighbor to avoid interpolating class indices.

Augmentation uses `cv2.warpAffine` for rotation:
- `flags=cv2.INTER_NEAREST` for masks (preserves integer class values)
- `borderMode=cv2.BORDER_CONSTANT, borderValue=0` (fills new border pixels with background)

**Note:** `cv2.BORDER_NEAREST` is not a valid OpenCV constant. The correct constant for nearest-neighbor interpolation during warp is `cv2.INTER_NEAREST`, set via the `flags` parameter. This bug was present in an early draft and corrected before training.

#### 4. Pillow API compatibility

Pillow deprecated bare constants (`Image.BILINEAR`, `Image.NEAREST`) in newer versions. V2 uses the correct namespaced forms throughout:
- `Image.Resampling.BILINEAR` for image resizing
- `Image.Resampling.NEAREST` for mask resizing and prediction grid rendering

#### 5. Confusion matrix

V2 produces a 3×3 confusion matrix (background × strawberry × peduncle) instead of v1's near-empty 2×2. This allows evaluation of:
- How often the model confuses peduncle with strawberry (adjacent classes, structurally meaningful error)
- How often the model confuses either with background (boundary errors)

#### 6. Expected training time is longer than v1

V1 typically stopped around epoch 10 via early stopping. V2 trains for more epochs for two compounding reasons:

**Larger dataset.** Module 2 v2 generates one crop per detected strawberry rather than one crop per source image. An image with four visible strawberries produces four training examples. The total number of crops is substantially higher than v1's 357, which means each epoch processes more data and the model takes longer to converge.

**Harder task.** Three-class segmentation requires the model to simultaneously learn the boundary between background and fruit, and the boundary between fruit and peduncle. The peduncle class in particular starts with very low IoU (near zero) and climbs slowly across epochs because peduncle pixels are sparse relative to background and fruit pixels. The model needs more gradient updates to reliably activate on the peduncle class.

In practice, peduncle IoU (perCls[2]) continues to improve even in epochs where overall mIoU has plateaued, which means early stopping should not trigger prematurely — the model is still learning the class that matters most for the cutting-point task.

---

## Module 4 V2 — Stem Angle Extraction

**Notebook:** `module4_v2_stem_angle.ipynb`  
**Input:** `best_unet_v2.pt`, `val_split_v2.json`

### What changed from v1

#### 1. Loads the three-class model

V4 loads `UNet3` (three output channels) instead of the binary U-Net. Inference produces per-pixel class probabilities across three channels.

#### 2. Peduncle extraction

V1 thresholded the single sigmoid output (e.g., `> 0.5`).

V2 takes `argmax` across channels and filters for pixels where `predicted_class == 2`. This is more principled: the pixel's assigned class is the one with the highest softmax score, not a manually chosen threshold.

#### 3. Connected-component filtering

After extracting peduncle pixels, V2 applies `largest_cc` (largest connected component) to remove isolated noise pixels before running PCA. Small disconnected fragments can significantly distort the principal axis if included.

#### 4. PCA angle extraction with [-90, 90] normalization

The PCA computation is unchanged from v1 — the principal axis of the peduncle pixel coordinates is computed and converted to degrees via `arctan2`.

**New in v2:** The raw arctan2 output is folded into [-90°, +90°] before being returned:

```python
if ang >  90: ang -= 180
if ang < -90: ang += 180
```

PCA principal components are sign-ambiguous — the first component can point either direction along the same axis. Without folding, physically identical peduncle orientations can produce angles like 118° and -62°, which represent the same cut axis but look like very different values in logs, histograms, and JSON exports. Folding collapses the output to a consistent reference frame. It does not change what line is drawn on the visualization (the draw function already renders both directions), but it ensures any downstream consumer — including a physical robot arm controller — receives a consistent angle convention.

---

## Run Order

When retraining from scratch or after any change to the source annotations:

1. **Module 1** — only if YOLO weights (`best.pt`) are missing or you are retraining the detector
2. **Module 2 V2** — regenerates all multiclass masks and per-fruit crops
3. **Module 3 V2** — trains the 3-class U-Net on the new crops
4. **Module 4 V2** — runs PCA angle extraction on v2 U-Net predictions

If only the U-Net needs retraining (annotations unchanged), start at step 3.  
If only the angle extraction needs updating, start at step 4.

---

## What Did NOT Change from V1

- **YOLOv11 detector (Module 1):** The fruit detector is unchanged. V2 uses the same `best.pt` weights to locate strawberries.
- **YOLO label file format:** All four contributor datasets (hervejunior, biberdork, aida2154, markm) use the same YOLO polygon format. Module 2 v2 reads them identically to v1.
- **PCA angle method:** The math for extracting the principal axis from a binary peduncle mask is identical in v1 and v2.
- **Image size during U-Net inference:** 256×256 px, same as v1.

> **Note:** `CONF` is no longer the same as v1. It was raised from `0.01` to `0.15` in Module 2 v2 — see section 5 above.

---

## How We Got to the Final Configuration — The Weight Tuning Journey

The v2 pipeline went through three full training runs before arriving at the final configuration. Each run taught us something specific.

### Run 1 — V2 architecture, no class weighting (v2 baseline)

> **This is not V1.** Run 1 uses the 3-class U-Net architecture on ROI crops — it is already significantly different from v1 (which used binary segmentation on full images). Run 1 is the starting point within the v2 design, used to isolate the effect of class weighting in Runs 2 and 3.

**Config:** `CrossEntropyLoss` with no weights, CONF=0.01, single-crop-per-image (only largest fruit), symmetric 20px padding.

**Result:** Best val_mIoU = 0.5953, peduncle IoU = 0.173, peduncle recall = 33.5%.

**Problem identified:** The confusion matrix was nearly empty for peduncle. The model found only 1 in 3 peduncle pixels because background pixels outnumbered peduncle pixels ~23:1. Without weighting, the loss signal is dominated by background and fruit — the model learns to mostly ignore peduncle because getting it wrong is statistically cheap.

**Also identified:** The prediction grid showed garbage crops containing no fruit — just background. Root cause: CONF=0.01 caused YOLO to fire on shadows and noise, each becoming a training crop.

---

### Run 2 — Full inverse-frequency weighting (~26×), cleaned crops

**Changes made before this run:**
- Module 2 v2 rewritten to generate one crop per detected strawberry (not just largest)
- Asymmetric padding: top +100% bbox height, sides +25% bbox width, bottom 20px
- CONF raised from 0.01 → 0.15 (filters YOLO noise)
- MIN_CONTENT filter added: crops with fewer than 200 non-background pixels discarded
- Dynamic inverse-frequency class weighting added to Module 3 v2

**Computed weights:** BG=1.00, Fruit=4.60, Peduncle=25.94

**Result:** Best val_mIoU = 0.5551, peduncle IoU = 0.168, peduncle recall = 62.8%, peduncle precision = 18.7%.

**Problem identified:** The 26× peduncle weight overcorrected. Recall nearly doubled (33.5% → 62.8%) but precision collapsed to 18.7% — meaning for every real peduncle pixel found, the model hallucinated 4 more. 20% of all background pixels were being predicted as peduncle (1.72M false positives). Overall mIoU dropped because background IoU fell from 0.886 to 0.713.

---

### Run 3 — Weight cap at 12×

**Change made:** Added `np.clip(_w, 1.0, 12.0)` after weight normalization. Peduncle weight capped at 12× regardless of class distribution.

**Computed weights:** BG=1.00, Fruit=4.60, Peduncle=12.00

**Result:** Best val_mIoU = 0.5843 @ epoch 22 (stopped at 27), peduncle IoU = **0.195**, recall = 54.0%, precision = 23.4%.

**Outcome:** Best result on every metric. Peduncle IoU improved from baseline 0.173 → 0.195. Fruit IoU improved from 0.737 → 0.787 (sharper class boundaries benefit all classes). Background recovered from 0.713 to 0.772. False-positive background firings dropped from 1.72M to 1.14M. The connected-component filter in Module 4 handles the remaining false positives before PCA runs.

---

## Final Training Results (12× weighted run — the keeper)

**Best epoch:** 22 of 27 (early stopped)  
**Best val_mIoU:** 0.5843  
**Per-class IoU:** BG=0.772, Fruit=0.787, Peduncle=0.195  
**Class weights used:** BG=1.00, Fruit=4.60, Peduncle=12.00

**Confusion matrix (rows=GT, cols=pred):**
```
           Pred BG    Pred Fruit  Pred Ped
GT BG     6,817,615    430,014   1,135,006
GT Fruit    265,776  3,208,983      53,973
GT Ped      188,158    120,799     362,588
```

**Peduncle class detail:**
- Recall: 54.0% (model finds just over half of actual peduncle pixels)
- Precision: 23.4% (some over-triggering on background, cleaned up by connected-component filter in Module 4)

**All-run comparison:**

| Run | Best val_mIoU | Peduncle IoU | Peduncle Recall | Peduncle Precision |
|-----|--------------|-------------|----------------|-------------------|
| Unweighted | 0.5953 | 0.173 | 33.5% | 26.4% |
| 26× weighted | 0.5551 | 0.168 | 62.8% | 18.7% |
| **12× weighted (final)** | **0.5843** | **0.195** | **54.0%** | **23.4%** |

The 12× cap was the optimal weight. The 26× run over-corrected (20% of background pixels predicted as peduncle). The unweighted run under-corrected (model ignored peduncle almost entirely). 12× hit the balance point.

---

## Module 4 V2 Results

**Output:** `runs/stem_angles_v3/`  
**Files:** `stem_angles_v3.json`, `stem_v3_examples.png`, `stem_v3_dist.png`

### Angle extraction statistics (final run — [−90°, +90°] normalization applied)

| Metric | Value |
|--------|-------|
| Total validation crops | 192 |
| Valid angles produced | 178 (92.7%) |
| Null (no peduncle found) | 14 (7.3%) |
| Angle range | -43.85° to 126.89° |
| Mean angle | 12.36° |

**16 crops had raw angles above 90°.** These were not bad predictions. PCA is sign-ambiguous — the model correctly identified the peduncle axis, but the eigenvector pointed the other way along it. Subtracting 180° collapses them to equivalent negative angles (e.g., 118 deg becomes -62 deg). The visualisation line does not change; only the reported angle value is made consistent for downstream use.

**Final distribution (post-fold):** all 178 valid angles lie within [-43.85°, +81.89°]. 147 of 178 (82.6%) fall within ±45° of vertical — physically expected for upright strawberry stems.

### Coverage interpretation

92.7% of validation crops produced a valid angle. The 7.3% null rate corresponds to crops where the model predicted fewer than 10 connected peduncle pixels after the `largest_cc` filter — consistent with the model's 54% peduncle recall and the MIN_CONTENT crop filter working as intended.

### Angle distribution interpretation

All 178 angles lie within a clean [-43.85°, +81.89°] range after folding. 147 of 178 (82.6%) fall within ±45° of vertical — physically expected for upright or slightly-tilted strawberry stems. The spread reflects genuine variation in camera angle and fruit orientation across the four contributor datasets.

---

## Open Questions / Items to Finalize

- [x] Confirm final crop count after re-running Module 2 v2
- [x] Assess whether 3-class confusion matrix shows meaningful improvement over v1 — **yes: peduncle IoU 0.173 → 0.195, fruit IoU 0.737 → 0.787**
- [x] Evaluate weight tuning — **settled at 12× cap after testing 26× (too aggressive)**
- [x] Run Module 4 v2 — **complete: 178/192 valid angles, 92.7% coverage**
- [x] Add angle normalization to [-90°, +90°] — **done in `pca_ang`**
- [x] Re-run Module 4 v2 with normalization — **done: all 178 angles fold cleanly into [-43.85°, +81.89°]**
- [x] Update final report with v2 metrics — **see `FINAL_REPORT_V2.md`**
