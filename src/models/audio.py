"""
audio.py – Inference pipeline for CityPulse acoustic classification and intensity.

This module provides the core runtime inference function `predict_audio(path)`
which loads an audio file, extracts acoustic features, evaluates classification
probabilities with the trained SVM-mixed candidate model, and computes the
calibrated relative audio intensity score.

Artifacts Loaded at Import
--------------------------
1. models/svm_mixed_candidate.joblib:
   Trained scikit-learn Pipeline (StandardScaler + SVC with probability=True)
   trained on clean and degraded UrbanSound8K folds 1–9.
2. models/intensity_bounds.joblib:
   Calibrated 5th (p5) and 95th (p95) percentile RMS dB bounds computed on
   unpadded training audio.
"""

from pathlib import Path
import numpy as np
import joblib

from src.preprocessing.audio import load_fixed, extract_features
from src.preprocessing.intensity import raw_db_unpadded

# Resolve paths relative to this file's own location (independent of working directory)
_MODULE_PATH = Path(__file__).resolve()
_REPO_ROOT = _MODULE_PATH.parent.parent.parent

_MODEL_PATH = _REPO_ROOT / "models" / "svm_mixed_candidate.joblib"
_BOUNDS_PATH = _REPO_ROOT / "models" / "intensity_bounds.joblib"

if not _MODEL_PATH.is_file():
    raise FileNotFoundError(f"Candidate model artifact not found at: {_MODEL_PATH}")

if not _BOUNDS_PATH.is_file():
    raise FileNotFoundError(f"Intensity bounds artifact not found at: {_BOUNDS_PATH}")

# Load model and intensity calibration parameters once at import
_model = joblib.load(_MODEL_PATH)
_bounds = joblib.load(_BOUNDS_PATH)

_p5 = float(_bounds["p5"])
_p95 = float(_bounds["p95"])


def predict_audio(path) -> dict:
    """Predict acoustic class, classification confidence, and relative intensity.

    Parameters
    ----------
    path : str or pathlib.Path
        Filesystem path to the input audio file.

    Returns
    -------
    dict
        A dictionary with keys:
        - "type" (str): Predicted acoustic class name among the 5 target classes.
        - "confidence" (float): Classification confidence score in [0.0, 1.0].
        - "relative_intensity" (float): Normalized intensity score in [0.0, 1.0].

    Raises
    ------
    ValueError
        If the audio file does not exist or is not a regular file.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise ValueError(f"Audio file does not exist or is not a valid file: {path}")

    # 1. Load waveform fixed to target duration/sampling rate and extract 128-d features
    y = load_fixed(file_path)
    feats = extract_features(y)

    # 2. Evaluate classification probabilities and predicted class
    feats_2d = feats.reshape(1, -1)
    proba = _model.predict_proba(feats_2d)[0]
    pred_type = str(_model.predict(feats_2d)[0])
    confidence = float(np.max(proba))

    # 3. Compute unpadded RMS dB and map to [0, 1] relative intensity
    db = raw_db_unpadded(file_path)
    rel_intensity = float(np.clip((db - _p5) / (_p95 - _p5), 0.0, 1.0))

    return {
        "type": pred_type,
        "confidence": confidence,
        "relative_intensity": rel_intensity,
    }

