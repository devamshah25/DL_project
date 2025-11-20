# Fine-Tuning DeiT for Image Classification and CLIP-Based Image Retrieval

## Overview

This project explores the capabilities of modern Vision Transformer models for two distinct computer vision tasks using a custom dataset of 10 everyday object classes:

1.  **Image Classification**: A DeiT-Tiny model (`deit-tiny-patch16-224`) was fine-tuned to classify images.
2.  **Image Retrieval**: A pre-trained CLIP model (`clip-vit-base-patch32`) was used to perform zero-shot, text-based image retrieval.

This repository contains the Jupyter notebook and all the necessary metadata files to reproduce the experiment.

## Repository Structure

This project is organized as follows:

*   `deit_tiny_DLproject-2.ipynb`: The main Jupyter Notebook containing all the code for data loading, training, evaluation, and analysis.
*   `labels_fixed.csv`: The core metadata file, mapping each image file to its corresponding class label.
*   `classes.txt`: A simple text file listing the 10 class names used in this experiment.
*   `attributes.yaml`: A file containing metadata about the attributes of the objects in the dataset.
*   `README.md`: This file, explaining the project.

## Models & Dataset

*   **Classification Model**: `facebook/deit-tiny-patch16-224`
*   **Retrieval Model**: `openai/clip-vit-base-patch32`
*   **Dataset**: A custom dataset of **10 classes** of everyday objects. The metadata is provided in this repository, but the image files themselves must be acquired separately.

## Key Features & Pipeline

*   **Data Loading**: Custom PyTorch `Dataset` and `DataLoader` for efficient data handling.
*   **Training**: A standard fine-tuning loop with the AdamW optimizer and Cross-Entropy Loss.
*   **Evaluation**: Comprehensive analysis including a classification report, a 10x10 confusion matrix, and t-SNE/PCA visualizations to understand model performance.
*   **Retrieval System**: A two-stage retrieval pipeline that first computes and saves CLIP embeddings for the validation set and then performs fast cosine similarity searches against text queries.

## Classification Results & Observations

The DeiT-Tiny model was fine-tuned for 12 epochs on the 10-class dataset, achieving the following performance on the validation set:

*   **Overall Accuracy**: `83%`
*   **Weighted F1-Score**: `0.8257`

#### Key Observations:
*   **Performance**: The model achieved high performance, demonstrating its ability to learn effectively on a focused dataset.
*   **Areas of Confusion**: The 10x10 confusion matrix showed minor confusion between `clothing_wrist_watch` and `electronics_accessories_phone_charger`. This is likely because `both classes have similar shape as in a central block connected to a long flexible line`.
*   **Feature Space**: The t-SNE plot for the 10-class dataset showed very distinct, well-separated clusters, indicating that the model could easily distinguish between these specific objects.

