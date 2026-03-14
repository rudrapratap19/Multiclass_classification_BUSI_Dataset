# Multiclass Classification on BUSI Dataset

This project presents a comparative deep learning study for multiclass breast ultrasound image classification using the BUSI dataset.

## Project Summary
The notebook implements and evaluates four training strategies with a ResNet-18 backbone:
- Baseline (CrossEntropyLoss)
- Data Augmentation
- Oversampling (WeightedRandomSampler)
- Focal Loss

The goal is to improve robustness on imbalanced classes: benign, malignant, and normal.

## Dataset
- BUSI Breast Ultrasound Dataset
- Total samples: 780 images
- Classes:
  - benign: 487
  - malignant: 210
  - normal: 133

## Repository Contents
- Busi_classification.ipynb: full experiment pipeline and analysis
- BUSI_classification_report.pdf: formal report of the project
- README.md: project overview and usage guide

## Tech Stack
- Python
- PyTorch, TorchVision
- NumPy, Pandas
- Matplotlib, Seaborn
- scikit-learn

## How to Run
1. Open the notebook in Jupyter or VS Code.
2. Update dataset path in the configuration cell if needed.
3. Run all cells in order.
4. Review experiment metrics, confusion matrices, and summary outputs.

## Evaluation Metrics
- Accuracy
- Macro F1-score
- Macro AUC
- Class-wise F1
- Confusion Matrix

## Notes
- The workflow uses fixed random seeds for reproducibility.
- A fixed train/test split is reused across all experiments for fair comparison.

## Author
Rudra Pratap
