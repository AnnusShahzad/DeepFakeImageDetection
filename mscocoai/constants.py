"""Shared constants for MS-COCOAI experiments (single source of truth)."""
import os

HF_DATASET_ID = "Rajarshi-Roy-research/Defactify_Image_Dataset"

# Paper Table: 42,000 / 9,000 / 45,000 images.
SPLIT_SIZES = {"train": 42000, "validation": 9000, "test": 45000}
SPLIT_ALIASES = {"val": "validation"}

# Label_B: 0 real, 1 SD2.1, 2 SDXL, 3 SD3, 4 DALL-E 3, 5 MJ6
LABEL_B_NAMES = ["real", "sd21", "sdxl", "sd3", "dalle3", "mj6"]
GENERATOR_NAMES = LABEL_B_NAMES[1:]
NUM_ATTR_CLASSES = 6
LABEL_A_NAMES = ["real", "ai-generated"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

IMG_SIZE = 224
SEED = 42

# Binary Label_A on the full benchmark is 1:5 (real:fake).
CLASS_FREQ_NOTE = "Label_A is 16k real vs 80k AI-generated (1:5); report macro metrics too."

def output_root() -> str:
    return os.environ.get("MSCOCOAI_OUTPUT_DIR", os.path.join(os.getcwd(), "outputs"))
