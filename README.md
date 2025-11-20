# Fine-Tuning DeiT for Image Classification and CLIP-Based Image Retrieval

## Overview

This project explores the capabilities of modern Vision Transformer models for two distinct computer vision tasks using a custom dataset of 10 everyday object classes:

1.  **Image Classification**: A DeiT-Tiny model (`deit-tiny-patch16-224`) was fine-tuned to classify images.
2.  **Image Retrieval**: A pre-trained CLIP model (`clip-vit-base-patch32`) was used to perform zero-shot, text-based image retrieval.

This repository contains the Jupyter notebook `deit_tiny_DLproject-2.ipynb`, which details the entire pipeline from data loading to analysis.

## Models & Dataset

*   **Classification Model**: `facebook/deit-tiny-patch16-224`
*   **Retrieval Model**: `openai/clip-vit-base-patch32`
*   **Dataset**: A custom dataset of **10 classes** of everyday objects, including `tableware_water_bottle`, `stationary_pen`, `clothing_wrist_watch`, etc. The dataset was split into training and validation sets using a stratified split to ensure representative class distribution.

## Key Features & Pipeline

*   **Data Loading**: Custom PyTorch `Dataset` and `DataLoader` for efficient data handling.
*   **Training**: A standard fine-tuning loop with the AdamW optimizer and Cross-Entropy Loss.
*   **Evaluation**: Comprehensive analysis including a classification report, a 10x10 confusion matrix, and t-SNE/PCA visualizations to understand model performance.
*   **Retrieval System**: A two-stage retrieval pipeline that first computes and saves CLIP embeddings for the validation set and then performs fast cosine similarity searches against text queries.

## Classification Results & Observations

The DeiT-Tiny model was fine-tuned for [Number] epochs on the 10-class dataset, achieving the following performance on the validation set:

*   **Overall Accuracy**: `[Your Overall Accuracy for the 10-class model]`
*   **Weighted F1-Score**: `[Your F1-Score for the 10-class model]`

#### Key Observations:
*   **Performance**: The model achieved high performance, demonstrating its ability to learn effectively on a focused dataset.
*   **Areas of Confusion**: The 10x10 confusion matrix showed minor confusion between `[Class A]` and `[Class B]`. This is likely because `[Your Reason, e.g., both classes share similar colors or shapes in the dataset]`.
*   **Feature Space**: The t-SNE plot for the 10-class dataset showed very distinct, well-separated clusters, indicating that the model could easily distinguish between these specific objects.

## How to Run

1.  Clone this repository.
2.  Install the required libraries (see `requirements.txt`).
3.  Open `deit_tiny_DLproject-2.ipynb` in a Jupyter environment like Google Colab.
4.  Make sure the file paths at the beginning of the notebook point to the correct location of your dataset in your Google Drive.
5.  Run the cells sequentially from top to bottom.

