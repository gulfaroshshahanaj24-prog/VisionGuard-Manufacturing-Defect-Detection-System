# Model Card — VisionGuard

## Intended use
Educational demonstration of binary image classification for manufacturing product images (`normal` vs `defective`).

## Not intended for
- Autonomous production-line rejection without human review
- Safety-critical or regulated quality decisions
- Claims of guaranteed defect detection
- Use outside the product categories and image conditions represented in training data

## Model
ResNet-18 pretrained on ImageNet (unless `--no-pretrained` is selected), with a two-class classification head. The final residual block and classifier head are trainable.

## Data
The project accepts a custom labeled dataset or one MVTec AD category prepared by `scripts/prepare_mvtec.py`. Record the dataset source, category, license/terms, collection conditions, class counts, and split strategy in the final report.

## Evaluation
Use the held-out test set. Report accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC. The internship brief's targets (>90% accuracy, >85% defective recall, >0.92 AUC) are goals, not guaranteed performance.

## Limitations
Performance can change with lighting, camera, product type, background, defect size, class imbalance, and distribution shift. MVTec AD masks are not used by this classifier. Confidence scores are not calibrated probabilities unless calibration is separately evaluated.

## Ethical and operational considerations
A human should review uncertain or high-impact cases. Do not conceal false negatives. Do not describe this educational model as a validated industrial inspection product.
