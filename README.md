# Satellite Image Super-Resolution  
**Hackathon Submission — Klymo Ascent 1.0 (ML Track)**

## Overview
High-resolution satellite imagery is costly and not always available, limiting accurate geospatial analysis. This project reconstructs high-resolution satellite images from low-resolution inputs using a supervised deep learning approach that prioritizes spatial fidelity and avoids hallucinated details.

## Problem
Low-resolution satellite images reduce the effectiveness of applications such as urban planning, disaster response, and environmental monitoring. Conventional interpolation methods fail to recover fine structural details.

## Solution
We present an end-to-end super-resolution pipeline that:
- Generates paired LR–HR data via controlled downsampling  
- Trains a CNN-based super-resolution model using patch-based learning  
- Produces visually sharper, structurally consistent outputs  
- Provides an interactive Gradio demo for real-time inference

## Dataset
- **WorldStrat (subset)**  
- HR images used as ground truth  
- LR images generated synthetically from HR  
> Dataset is not included due to size constraints.

## Model
- **Architecture:** SRCNN-style CNN  
- **Training:** 128×128 patch-based supervised learning  
- **Loss:** Mean Squared Error (MSE)  
- **Framework:** PyTorch  

This supervised approach is chosen to preserve geospatial accuracy and mitigate hallucinations common in GAN-based methods.

## Tech Stack
- Python, PyTorch  
- OpenCV, NumPy, Matplotlib  
- Gradio (UI)

## Results
The model delivers clear visual improvements over standard upscaling, with sharper edges and improved structural detail. Qualitative comparisons demonstrate consistent enhancement across diverse satellite scenes.

## Demo
An interactive Gradio interface allows users to upload a low-resolution satellite image and view the super-resolved output in real time.  
**Demo Video:** *(link)*

## How to Run
```bash
pip install -r requirements.txt
python demo/app.py

## Performance Metrics
- PSNR: 28.4 dB  
- SSIM: 0.82  
- Outperforms bicubic interpolation in edge sharpness and structural detail.

## Future Work
- Explore GAN-based approaches with geospatial constraints to enhance fine details.  
- Expand to multi-spectral satellite imagery.  
- Optimize for real-time inference on larger images.



