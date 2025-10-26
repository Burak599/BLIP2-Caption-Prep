# BLIP2 Captioning Prep

Automatic caption generation for images using BLIP-2.  
Prepares datasets with captions and LoRA tokens for fine-tuning.

## Features
- Generate captions for images automatically.
- Append custom LoRA tokens and classes for fine-tuning.
- Saves captions as `.txt` files next to images.
- Supports GPU (float16) and CPU (float32).

## Installation
```bash
git clone https://github.com/Burak599/BLIP2-Caption-Prep.git
cd BLIP2-Caption-Prep
pip install -r requirements.txt
```

## Data Structure
dataset/
└── train/
    ├── image1.jpg
    └── image2.jpg
