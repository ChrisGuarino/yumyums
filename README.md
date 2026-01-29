# YumYums (Catcha v2)

A motion-activated cat detection and identification system using a fine-tuned Vision Transformer (ViT) to classify individual cats from a live camera feed.

## Overview

The system monitors a camera feed using OpenCV frame differencing to detect motion. When movement is detected, a fine-tuned ViT model classifies whether the frame contains a specific cat (Prim or Rupert) or no cat at all.

## How It Works

1. **Motion detection** — Frame differencing between consecutive camera frames identifies movement
2. **Classification** — When motion is triggered, the ViT model runs inference on the current frame
3. **Identification** — The model outputs one of: Prim, Rupert, or no cat

## Model

- **Base model:** [google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k)
- **Fine-tuned model:** [ChrisGuarino/model](https://huggingface.co/ChrisGuarino/model)
- **Training dataset:** [ChrisGuarino/cat_ds](https://huggingface.co/ChrisGuarino/cat_ds)

## Setup

```bash
./setup/setup_env.sh
```

## Usage

Run the cells in `main/main.ipynb` to start the camera feed with detection and classification.

## Project Structure

```
yumyums/
├── main/
│   ├── main.ipynb           # Main notebook — camera + detection + classification
│   ├── ModelTrain.ipynb     # Model training notebook
│   └── ModelCameraTest.ipynb# Camera integration testing
├── setup/
│   └── setup_env.sh         # Environment setup script
├── ImageHolder.py           # Camera capture utility class
├── change_detection.ipynb   # Frame differencing experiments
├── dataset_raw/             # Raw training images
└── README.md
```
