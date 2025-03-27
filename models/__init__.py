from models.dust3r.model import AsymmetricCroCo3DStereo
from models.dust3r.dyna_dust3r import DynaDUSt3R

MODELS = {
    "DUSt3R": AsymmetricCroCo3DStereo,
    "DynaDUSt3R": DynaDUSt3R,
}

def get_model(config):
    model_name = config.model.name
    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not found in available models: {MODELS.keys()}")
    model = MODELS[model_name](config)

    return model