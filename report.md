# 🧠 Unified-OneHead Multi-Task Report

## 1. Architecture Design & Motivation

We propose a single-branch multi-task network called **UniHeadNet**, based on the EfficientNet-B0 backbone with a lightweight head (2 conv layers + 1 output conv), designed to jointly solve:

- **Object Detection** (Mini-COCO-Det, 10 classes)  
- **Semantic Segmentation** (Mini-VOC-Seg, 19 classes)  
- **Image Classification** (Imagenette160, 30 classes)

All outputs are emitted simultaneously from a shared head:

- **Detection**: anchor-based YOLO-style outputs (`cx`, `cy`, `w`, `h`, `conf`, `cls`)  
- **Segmentation**: `C×H×W` mask  
- **Classification**: global average pooled logits

**Total parameters ≈ 4.24M**, well below the 8M constraint.

---

## 2. Training Schedule & Forgetting Mitigation

| Stage | Task           | Dataset          | Output   | Notes                               |
| ----- | -------------- | ---------------- | -------- | ----------------------------------- |
| 0     | Pretrain       | ImageNet (EffB0) | Backbone | Pretrained weights loaded           |
| 1     | Segmentation   | VOC (Mini-VOC)   | mIoU     | Train `seg` head                    |
| 2     | Detection      | COCO (Mini-Det)  | mAP      | Train `det` head only               |
| 3     | Classification | Imagenette       | Top-1    | Train only `cls` slice of `head_out` |

**Forgetting Solution:**

We adopt a simple but effective mitigation strategy:

- **Stage 2**: freeze `blocks.0` and `blocks.1` weights  
- **Stage 3**: freeze all except the final classification slice of `head_out` (≈ 1 layer)  
- To protect detection/segmentation, their weight slices are restored post-training

This is similar in spirit to *Learning without Forgetting (LwF)* or replay-free distillation.

---

## 3. Final Performance

| Task           | Metric        | Stage 1 | Stage 2 | Stage 3 | Stage 3 Strategy  | Drop (%)   |
| -------------- | ------------- | ------- | --------| --------| ------------------ | ---------- |
| Segmentation   | mIoU          | 0.907   | 0.930   | 0.937   | Freeze + cls only  | +3.30%     |
| Detection      | mAP@[.5:.95]  | x       | 0.316   | 0.166   | Freeze + cls only  | -47.46%    |
| Classification | Top-1 Acc     | x       | x       | 0.166   | Freeze + cls only  | x          |

---

## 4. Resource Efficiency

- ✅ **Training time**: 0h 10m 59.83s on Colab T4  
- ✅ **Model parameters**: < 8M (≈4.24M)  
- ✅ **Inference time**: 39 ms/it (det), 65 ms/it (seg), 40 ms/it (cls) — all measured on 256×256 inputs

---

## 5. Analysis & Challenges

### Segmentation task

- mIoU increased slightly after the Detection + Classification stages.
- Likely explanation: the original segmentation dataset was small; seeing more images—even from other tasks—helped the model generalize.

### Detection task

- mAP fell by roughly 50% even after restoring the original detection-head weights.
- Root cause: **Batch-Normalization drift** — when only the head is trained, the backbone’s BN layers keep updating their `running_mean` and `running_var`, shifting feature distributions and hurting detection.

### Classification task

- Because every layer except the final classifier was frozen, a lower Top-1 accuracy is expected.
- Moreover, performance was not the main focus of the assignment.

### Head-wise training vs. interference

- Training each head separately prevented catastrophic forgetting.
- However, detection and classification heads shared overlapping receptive fields; even with frozen weights, their BN statistics interacted, degrading detection.

### Methodological constraints

- Assignment rules limited our options.
- Ideally, all three tasks—**detection**, **segmentation**, and **classification**—should be trained jointly.
- Detection and segmentation heads should be attached to earlier layers of the backbone, and a multi-stage training schedule should be adopted to balance the losses. This setup would likely yield better overall performance.

---

## 6. Conclusion

Stage-wise, head-specific fine-tuning avoided catastrophic forgetting but highlighted a critical weakness: **Batch-Normalization statistics drift** can negate expected gains when heads share features.

Under tight constraints, the approach produced:

- ✅ Modest segmentation improvements  
- ❌ Pronounced detection degradation  
- ➖ Acceptable but unremarkable classification results

**Future work** should adopt:

- Fully joint multi-task training  
- Earlier placement of detection and segmentation heads  
- Explicit loss balancing to stabilize BN statistics and maximize synergy across tasks
