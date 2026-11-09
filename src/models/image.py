"""
CityPulse - image classifier interface (Part 2, Bushra Zafar)

    from src.models.image import predict_image
    predict_image("demo_samples/images/pothole_01.jpg")
    -> {"type": "pothole", "confidence": 0.93}

Interface contract (fixed):  predict_image(image_path) -> {type, confidence}
    type        one of the classes the model was trained on:
                "pothole", "crack", "normal", "flood"   (no "flood" if trained with --skip-flood)
    confidence  float in [0, 1]  = highest class probability (SVM: Platt-scaled, RF: vote share).
                Treat it as a relative score, not a calibrated guarantee.

Pipeline (identical to training):
    RGB -> resize 224x224 -> ImageNet mean/std -> MobileNetV2 (classifier removed, 1,280-d pooled
    output, CPU) -> StandardScaler+SVM / RandomForest from models/image_model.joblib
The MobileNetV2 weights are stored INSIDE the joblib, so no internet / torch cache is needed at demo time.
"""
import os
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_MODEL_PATH = Path(os.environ.get("CITYPULSE_IMAGE_MODEL", _ROOT / "models" / "image_model.joblib"))
_CACHE = {}


def _load():
    if "bundle" in _CACHE:
        return _CACHE
    import joblib
    import torch
    from torchvision import models, transforms

    if not _MODEL_PATH.exists():
        raise FileNotFoundError(f"image model not found: {_MODEL_PATH} (run src/image/s6_train_models.py)")
    bundle = joblib.load(_MODEL_PATH)

    net = models.mobilenet_v2(weights=None)
    net.classifier = torch.nn.Identity()
    net.load_state_dict({k: torch.from_numpy(np.asarray(v)) for k, v in bundle["backbone_state"].items()})
    net.eval()

    size = int(bundle["img_size"])
    tfm = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(bundle["mean"], bundle["std"]),
    ])
    _CACHE.update(bundle=bundle, net=net, tfm=tfm)
    return _CACHE


def _proba(image_path):
    import torch
    from PIL import Image

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")
    c = _load()
    with Image.open(image_path) as im:
        x = c["tfm"](im.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        emb = c["net"](x).numpy().astype(np.float32)
    est = c["bundle"]["estimator"]
    return est.classes_, est.predict_proba(emb)[0]


def predict_image(image_path):
    """Classify one photo. Returns {'type': str, 'confidence': float}."""
    classes, p = _proba(image_path)
    i = int(np.argmax(p))
    return {"type": str(classes[i]), "confidence": round(float(p[i]), 4)}


def predict_image_proba(image_path):
    """Extra helper (not part of the contract): full class-probability dict for the explainability panel."""
    classes, p = _proba(image_path)
    return {str(c): round(float(v), 4) for c, v in zip(classes, p)}


if __name__ == "__main__":
    import sys
    for f in sys.argv[1:]:
        print(f, predict_image(f))
