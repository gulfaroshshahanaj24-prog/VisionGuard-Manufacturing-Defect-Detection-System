# Dataset Guide

## Accepted custom dataset layout

```text
data/raw/
  normal/
  defective/
```

Place only labeled product images in the corresponding class. Use varied lighting, angles, product instances, and defect types where possible. Avoid duplicate or near-duplicate images.

## MVTec AD

Official page: https://www.mvtec.com/company/research/datasets/mvtec-ad

MVTec AD is organized by product/texture category. The preparation utility in this project uses:
- `train/good` as normal
- `test/good` as normal
- each defect-type directory under `test/` as defective

It does not use the pixel-level ground-truth masks. This project is image-level binary classification, not anomaly localization or segmentation.

## Data leakage warning

The split utility randomly partitions images into train, validation, and test sets. If several images come from the same physical item, production batch, or near-identical capture sequence, random splitting can overestimate generalization. Where possible, split by product instance or acquisition group before training and document that method.

## Class balance

Check the number of normal and defective examples. A model can show high accuracy while missing defects when classes are imbalanced. Always report defective-class recall and the confusion matrix alongside accuracy.
