"""
Author: Kevin Mathew T
Date: 2025-08-17
LinkedIn: https://www.linkedin.com/in/kevinmathewt/
"""

from models.dust3r.dyna_dust3r import DynaDUSt3R

MODELS = {
    "DynaDUSt3R": DynaDUSt3R,
}

def get_model(config, device):
    model_name = config.model.name
    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not found in available models: {MODELS.keys()}")

    return MODELS[model_name].load_model(config, device)
