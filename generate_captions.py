import torch
import os
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

torch.cuda.empty_cache()

# SETTINGS
# The dataset path must be correct
DATASET_PATH = r"path/to/your/dataset"

# Custom LoRA tokens for fine-tuning
LORA_TOKEN = "[custom_style]"  # Unique identifier for your dataset's concept or style (e.g. [anime_face], [modern_building])
LORA_CLASS = "object"  # Broad category of the images (e.g. person, building, vehicle)

# MODEL LOADING
MODEL_NAME = "your_model_name_here"  # e.g., "Salesforce/blip2-flan-t5-xl"
# Replace with a model suitable for your GPU capacity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == 'cpu':
    print("WARNING: Device set to CPU. Labeling will be VERY SLOW.")
else:
    print(f"Models are being loaded on GPU ({device})...")

try:
    print(f"Loading BLIP-2 {MODEL_NAME} processor...")
    # Automatically loads the correct tokenizer and image processor
    blip_processor = Blip2Processor.from_pretrained(MODEL_NAME)

    print(f"Loading BLIP-2 {MODEL_NAME} model (this may take a while on first run)...")
    # Use float16 to reduce VRAM usage
    blip_model = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)

    print("Model successfully loaded.")

except Exception as e:
    print(f"CRITICAL ERROR while loading BLIP2 model: {e}")
    print("Ensure that your torch and transformers libraries are up to date, and the model name is correct.")
    exit()

# DATA LOADING
try:
    print("\nLoading dataset (checking for train/ folder)...")
    dataset = load_dataset("imagefolder", data_dir=DATASET_PATH)["train"]
    print(f"Found {len(dataset)} images in the dataset.")
except KeyError:
    print("\nCRITICAL ERROR: 'train' subfolder not found!")
    print(f"Please create a folder named 'train' inside '{DATASET_PATH}' and move all your images there.")
    exit()

# CAPTION GENERATION FUNCTION 
def generate_captions(examples):
    # Convert PIL images to RGB
    images = [img.convert("RGB") for img in examples["image"]]

    # Generate captions using BLIP-2
    inputs = blip_processor(images=images, return_tensors='pt').to(device, torch.float16)

    # Generate short captions (max 77 new tokens)
    out = blip_model.generate(**inputs, max_new_tokens=77)

    # Decode model outputs into text
    captions = blip_processor.batch_decode(out, skip_special_tokens=True)

    # Append the custom LoRA token and class to each caption
    examples["text"] = [f"{c.strip()}, {LORA_TOKEN} {LORA_CLASS}" for c in captions]

    # Keep the original image file path
    examples["image_path"] = examples["image"].path

    return examples

# RUN CAPTIONING
print("\nStarting automatic caption generation (this may take some time)...")

# Run captioning in batches
dataset_with_captions = dataset.map(
    generate_captions,
    batched=True,
    remove_columns=["image"],
    desc="Generating captions for images",
    batch_size=2  # Number of images processed at once; increase if GPU memory allows, decrease if out-of-memory errors occur
)

# SAVE CAPTIONS
print("Saving captions as .txt files next to the images...")
for i, item in tqdm(enumerate(dataset_with_captions), total=len(dataset_with_captions), desc="Saving files"):
    # Save a .txt file next to each image with the same name
    image_path = item["image_path"]
    txt_path = os.path.splitext(image_path)[0] + ".txt"

    try:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(item["text"])
    except Exception as e:
        print(f"WARNING: Could not save {txt_path}. Error: {e}")

print("\nAutomatic captioning completed!")
print(f"Created .txt caption files (with LoRA tokens) for all {len(dataset_with_captions)} images.")
print("You can now proceed to run your main training script.")