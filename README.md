# Feature Representations & Robustness in Image Classification

A comparative machine learning study investigating how feature representations influence classification accuracy, robustness, and generalization under distribution shifts.

**University of Toronto | CSCC11 — Introduction to Machine Learning**

## Overview

This project evaluates three image feature representations—raw pixels, Histogram of Oriented Gradients (HOG), and pretrained CLIP embeddings—using a balanced binary classification dataset of 1,000 grayscale muffin and puppy images.

Three classifiers are evaluated: K-Nearest Neighbors, Random Forest, and Logistic Regression.

## Key Results

- **99.0% clean validation accuracy** using CLIP ViT-B/32 embeddings with Logistic Regression.
- CLIP retained **95.5% accuracy** under 30° image rotation with Logistic Regression.
- HOG demonstrated strong robustness to global contrast changes but was more sensitive to rotation.
- CLIP-based classifiers maintained **97.5–99.0% validation accuracy** under up to 30% randomly corrupted training labels after hyperparameter tuning.

## Methodology

- Dataset: 1,000 grayscale images, 64 × 64 pixels
- Split: 800 training / 200 validation images
- Feature representations: Pixels, HOG, CLIP ViT-B/32
- Classifiers: KNN, Random Forest, Logistic Regression
- Evaluation: Hyperparameter sweeps, training and validation accuracy, distribution shift, label noise

## Experiments

1. **Feature Representation Comparison:** Evaluate how different representations affect classification accuracy.
2. **Contrast Reduction:** Test robustness under simulated illumination changes.
3. **Image Rotation:** Evaluate geometric robustness using 30° rotations.
4. **Training Label Corruption:** Study the effect of randomly flipped labels and regularization.

## Technologies

Python · NumPy · Scikit-learn · PyTorch · OpenAI CLIP · HOG · Matplotlib

## Technical Report

See [Full Technical Report](reports/C11_A3_programming.pdf).

## Main Takeaway

Feature representation was the most influential factor in classification performance in this experiment. Pretrained semantic embeddings improved both accuracy and robustness compared with raw pixel features and handcrafted HOG descriptors.