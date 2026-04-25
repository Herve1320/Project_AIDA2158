# Phase 2 — YOLOv11-seg: Strawberry Detection & Segmentation Training
**AIDA 2158A Final Project | Mark Miller**

---

## What This Phase Did

Trained a YOLOv11 segmentation model to detect and outline individual strawberries in RGB images. This is Module 1 of the project specification.

---

## Dataset Used

| Split | Images | Labels | Source |
|---|---|---|---|
| Train | 2,800 (1,496 + 1,304) | 2,800 polygon `.txt` files | Google Drive dataset (pre-labelled) |
| Val | 100 | 100 (converted from instance PNG maps — see below) | Google Drive dataset |
| Test | 200 | 200 polygon `.txt` files | Google Drive dataset |
| **Total** | **3,100** | **3,100** | |

**Important note on val labels:** The val set did not include YOLO `.txt` label files — only `labels_non_binary/` PNG images where each pixel value represents a strawberry instance ID. We converted these automatically using contour extraction (OpenCV `findContours`) to produce valid YOLO polygon format labels before training.

---

## Model

**`yolo11s-seg.pt`** — the small YOLOv11 segmentation model, pretrained on COCO.

This was the model recommended by the project specification. "Small" means faster training while still achieving strong accuracy. It started from pretrained COCO weights (transfer learning), which is why it converged quickly and to a high accuracy.

---

## Hyperparameters

| Parameter | Value | Reasoning |
|---|---|---|
| Epochs | 50 (ran 48) | Sufficient to converge; early stopping triggered at epoch 48 |
| Image size | 640 × 640 | Standard YOLO input; balances speed vs accuracy |
| Batch size | 8 | Fits RTX 5070 VRAM; stable gradient updates |
| Optimizer | AdamW | Better convergence than SGD for segmentation; YOLO11 default |
| Learning rate | Auto (cosine annealed from 0.01) | YOLO handles LR scheduling internally |
| Early stopping patience | 15 | Stops if val mAP doesn't improve for 15 consecutive epochs |
| Device | GPU 0 (RTX 5070) | CUDA-accelerated training |
| Pretrained | Yes | Transfer learning from COCO speeds up convergence |

---

## Results

| Metric | Value | Interpretation |
|---|---|---|
| **mAP50 (box)** | **0.9273** | Excellent — model finds strawberries with high overlap |
| **mAP50-95 (box)** | **0.7696** | Strong — holds up at stricter IoU thresholds |
| **mAP50 (mask/seg)** | **0.9179** | Excellent — segmentation masks are accurate |
| **mAP50-95 (mask)** | **0.7011** | Good — masks remain precise at stricter thresholds |
| **Precision** | **0.8877** | When the model says "strawberry", it's right 89% of the time |
| **Recall** | **0.8439** | The model finds 84% of all strawberries in a scene |

Training converged at epoch 48 (early stopping). Train and val losses tracked closely throughout — **no significant overfitting**.

---

## Why These Results Are Good

- mAP50 above 0.90 is generally considered excellent for object detection
- The segmentation mAP (0.918) means the mask outlines are very accurate, not just the bounding boxes
- Precision and recall are balanced — the model is neither over-detecting nor missing too many
- Train vs val loss gap is small — the model generalises well to unseen images

---

## Files in This Folder

| File | Description |
|---|---|
| `NOTES.md` | This file |
| `module1_yolov11_training.ipynb` | Full training notebook with all code and comments |
| `results.csv` | Per-epoch metrics (loss, mAP, precision, recall) |
| `results.png` | YOLO auto-generated results grid |
| `training_curves.png` | Custom loss + mAP plots per epoch |
| `confusion_matrix.png` | Confusion matrix (raw counts) |
| `confusion_matrix_normalized.png` | Confusion matrix (normalised %) |
| `BoxPR_curve.png` | Precision-recall curve for bounding boxes |
| `BoxF1_curve.png` | F1 score curve for bounding boxes |
| `MaskPR_curve.png` | Precision-recall curve for segmentation masks |
| `MaskF1_curve.png` | F1 score curve for segmentation masks |
| `labels.jpg` | Distribution of label locations across the dataset |
| `val_batch0_labels.jpg` | Ground truth masks on val batch 0 |
| `val_batch0_pred.jpg` | Model predictions on val batch 0 |
| `val_batch1_labels.jpg` | Ground truth masks on val batch 1 |
| `val_batch1_pred.jpg` | Model predictions on val batch 1 |
| `val_predictions_sample.png` | Custom 6-image prediction overlay grid |
| `train_batch0.jpg` | Sample augmented training batch |
| `args.yaml` | Exact training configuration (all parameters) |

---

## Presentation Talking Points

1. We trained YOLOv11-seg from pretrained COCO weights on 2,800 strawberry images
2. The model achieved **92.7% mAP50** on detection and **91.8% mAP50** on segmentation
3. Training took approximately 48 epochs with early stopping — converged cleanly
4. No overfitting — training and validation losses stayed close throughout
5. Show: `training_curves.png`, `val_predictions_sample.png`, `confusion_matrix_normalized.png`
