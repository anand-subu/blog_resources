# config.py
import torch

# Model configuration
MODEL_TYPES = {
    "owlvit": "google/owlvit-base-patch32",
    "grounding_dino": "IDEA-Research/grounding-dino-tiny",
}

INV_MODEL_TYPES = {v:k for k,v in MODEL_TYPES.items()}

DEFAULT_MODEL_TYPE = "grounding_dino"
MODEL_ID = MODEL_TYPES[DEFAULT_MODEL_TYPE]

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Visualization settings
COLOR_PALETTE = [
    "red", "blue", "green", "purple", "orange", 
    "cyan", "magenta", "yellow", "brown", "pink"
]

# Threshold settings
CONFIDENCE_THRESHOLD = 0.2

OPENAI_API_KEY = ""
