# Fine-Tuning DeiT for Image Classification and CLIP-Based Image Retrieval

## Overview

This project explores the capabilities of modern Vision Transformer models for two distinct computer vision tasks:
1.  **Image Classification**: A DeiT-Tiny model (`deit-tiny-patch16-224`) was fine-tuned to classify images from custom datasets of everyday objects.
2.  **Image Retrieval**: A pre-trained CLIP model (`clip-vit-base-patch32`) was used to perform zero-shot, text-based image retrieval on the same datasets.

This repository contains the two main Jupyter notebooks used for these experiments: `deit_tiny_DLproject-2.ipynb` for the initial 10-class dataset and `deit_tiny_pooled-4.ipynb` for the final 197-class dataset.

## Models & Datasets

*   **Classification Model**: `facebook/deit-tiny-patch16-224`
*   **Retrieval Model**: `openai/clip-vit-base-patch32`
*   **Datasets**:
    *   **Initial Dataset**: A small, focused dataset of **10 classes** used for initial prototyping and analysis.
    *   **Final Dataset**: A larger, pooled dataset (`DL-2025-Dataset`) containing **197 classes**, used for the final model training and evaluation.

## Key Features & Pipeline

*   **Data Loading**: Custom PyTorch `Dataset` and `DataLoader` for efficient data handling, including the use of **stratified splitting** to ensure representative class distribution in the training and validation sets.
*   **Training**: A standard fine-tuning loop with the AdamW optimizer and Cross-Entropy Loss.
*   **Evaluation**: Comprehensive analysis including classification reports, confusion matrices, and t-SNE/PCA visualizations to understand model performance.
*   **Retrieval System**: A two-stage retrieval pipeline that first computes and saves CLIP embeddings for the validation set and then performs fast cosine similarity searches against text queries.

---

## Experiment 1: 10-Class Dataset Results

The initial experiment focused on a small set of 10 distinct classes to establish a performance baseline.

*   **Overall Accuracy**: `[Your Accuracy for the 10-class model]`
*   **Weighted F1-Score**: `[Your F1-Score for the 10-class model]`

#### Key Observations (10 Classes):
*   **Performance**: The model achieved high performance quickly, demonstrating its ability to learn effectively even on a small scale.
*   **Areas of Confusion**: The confusion matrix showed minor confusion between `[Class A from 10-class set]` and `[Class B from 10-class set]`, likely due to `[Your Reason, e.g., similar colors or shapes]`.
*   **Feature Space**: The t-SNE plot for the 10-class dataset showed very distinct, well-separated clusters, indicating that the model could easily distinguish between these specific objects.

---

## Experiment 2: 197-Class Dataset Results

The final experiment scaled up the challenge by using the full dataset of 197 classes.

*   **Overall Accuracy**: `[Your Overall Accuracy for the 197-class model, e.g., 0.85]`
*   **Weighted F1-Score**: `[Your Weighted F1-Score for the 197-class model]`

#### Key Observations (197 Classes):
*   **Scalability**: While the overall accuracy was strong, it was lower than the 10-class experiment, which is expected given the significantly increased complexity of the task.
*   **Strongest Classes**: The model continued to perform exceptionally well on classes with unique visual features, such as `[Your Best Performing Class from 197-class set]`, achieving an F1-score of `[Score]`.
*   **Areas of Confusion**: With more classes, new and interesting confusions emerged. The model most frequently confused `[Class X from 197-class set]` with `[Class Y from 197-class set]`. This is likely because `[Your Reason, e.g., both are small, metallic objects with similar textures]`.
*   **Feature Space**: The t-SNE plot for the 197-class dataset was naturally more crowded. However, it still revealed clear groupings of related super-categories (e.g., all `personal_care` items clustering loosely together), while showing significant overlap for the most commonly confused classes.

## Image Retrieval Observations

The CLIP-based retrieval system was tested on the larger dataset and demonstrated strong zero-shot capabilities:

*   **Specific Queries**: For precise queries like `"a new red plastic pen"`, the model was highly effective at retrieving relevant images.
*   **Ambiguous Queries**: For broader queries like `"something to write with"`, the model returned a mix of objects, including `[Example 1]`, `[Example 2]`, and `[Example 3]`. This highlights its ability to understand semantic relationships but also its potential to make logical but incorrect associations.

## How to Run

1.  Clone this repository.
2.  Ensure you have the required libraries installed (e.g., `torch`, `transformers`, `scikit-learn`, `matplotlib`, `seaborn`).
3.  Open either `deit_tiny_DLproject-2.ipynb` (10-class experiment) or `deit_tiny_pooled-4.ipynb` (197-class experiment) in a Jupyter environment like Google Colab.
4.  Make sure the file paths at the beginning of the notebook point to the correct location of your dataset in your Google Drive.
5.  Run the cells sequentially from top to bottom.

