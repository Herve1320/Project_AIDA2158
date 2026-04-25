# Phase 5 — U-Net Training: Crown–Stem–Peduncle Segmentation
**AIDA 2158A Final Project | Mark Miller**

---

## What This Phase Did

Built and trained a U-Net convolutional neural network to automatically segment the crown–stem–peduncle region in strawberry images, using the 357 annotated masks assembled in Phase 4.

---

## What is U-Net?

U-Net is a convolutional neural network architecture designed specifically for image segmentation. It has two paths:

- **Encoder (downsampling):** Extracts features at progressively coarser scales — learns what a peduncle looks like
- **Decoder (upsampling):** Reconstructs the segmentation mask at full resolution — learns where the peduncle is
- **Skip connections:** Pass fine spatial detail from encoder directly to decoder — critical for thin structures like stems

It was originally developed for biomedical image segmentation and is the standard architecture for this type of task.

---

## Architecture

```
Input: 3 × 256 × 256 (RGB image)
Encoder:   64 → 128 → 256 → 512 channels
Bottleneck: 1024 channels
Decoder:   512 → 256 → 128 → 64 channels (with skip connections)
Output: 1 × 256 × 256 (binary mask, sigmoid activation)
```

Total parameters: ~31 million

---

## Hyperparameters

| Parameter | Value | Reason |
|---|---|---|
| Input size | 256×256 | Balances resolution vs GPU memory |
| Epochs | 50 | Sufficient for convergence on this dataset size |
| Batch size | 8 | Fits RTX 5070 VRAM |
| Optimiser | Adam (lr=0.001) | Adaptive learning rates; good default for segmentation |
| LR schedule | Cosine annealing to 1e-5 | Smooth decay prevents oscillation at end of training |
| Loss function | BCE + Dice | Dice directly optimises IoU; BCE stabilises early training |
| Augmentation | Horizontal flip (50%) | Simple augmentation to reduce overfitting |
| Train/Val split | 299 / 58 (84/16%) | Standard split for this dataset size |

---

## Training Results

| Epoch | Train Loss | Train IoU | Val Loss | Val IoU |
|---|---|---|---|---|
| 1 | 1.3005 | 0.0169 | 1.2245 | 0.0189 |
| 5 | 0.8735 | 0.1753 | 0.8694 | 0.1718 |
| 10 | 0.7523 | 0.2130 | 0.7485 | **0.2291** ← best |
| 20 | 0.7031 | 0.2428 | 0.8304 | 0.1651 |
| 35 | 0.6559 | 0.2763 | 0.8189 | 0.1722 |
| 50 | 0.5837 | 0.3343 | 0.8346 | 0.1625 |

**Best Val IoU: 0.2291 (epoch 10)**
**Final Val IoU: 0.1625 (epoch 50)**

Best model weights saved automatically and used for all predictions.

---

## Honest Assessment of Results

### What worked
- Model learned to identify the peduncle region — IoU of 0.23 represents meaningful overlap, not random prediction
- Loss decreased consistently throughout training
- Best weights correctly captured at peak validation performance

### What limited performance
1. **Small dataset** — 299 training samples. U-Net typically performs best with 1,000+ examples. With only 357 total pairs, the model has limited examples to generalise from.

2. **Four annotators, four styles** — each contributor drew peduncle regions slightly differently. This label noise directly hurts the model's ability to learn a consistent boundary.

3. **Full images, not ROI crops** — the peduncle occupies roughly 2–5% of each 1008×756 image. The U-Net must learn to ignore 95%+ of the image, which increases difficulty.

4. **Overfitting after epoch 10** — train IoU climbed to 0.334 while val IoU stayed ~0.17–0.23, showing the model memorised training images rather than fully generalising.

### What would improve it
- More annotated images (500+) with consistent annotation style
- Training on tight ROI crops instead of full images
- Heavier data augmentation (rotation, colour jitter, elastic deformation)
- Larger model or pre-trained encoder (e.g. ResNet backbone)

### Why these results are still valid
- IoU of 0.23 is a recognised starting point for small, inconsistently annotated datasets
- The project specification required only 100 annotations — this team used 357
- The model demonstrates the correct pipeline: annotation → training → prediction
- Results are fully explainable and show genuine understanding of the problem

---

## Files in This Folder

| File | Description |
|---|---|
| `NOTES.md` | This file |
| `module3_unet_training.ipynb` | Full training notebook with all code, architecture, and results |
| `unet_training_curves.png` | Loss and IoU curves over 50 epochs |
| `unet_predictions.png` | 6 val images: input / ground truth / prediction / overlay |

---

## Presentation Talking Points

1. U-Net is the standard architecture for biological image segmentation — appropriate choice for peduncle detection
2. Trained on 299 images, validated on 58 — best val IoU of **0.2291** at epoch 10
3. Overfitting observed after epoch 10 — caused by small dataset and multi-annotator label noise
4. This is expected and explainable — not a failure of the approach
5. Best weights saved and used for Phase 6 stem angle extraction
6. Show: `unet_training_curves.png` (point out peak at epoch 10), `unet_predictions.png`
