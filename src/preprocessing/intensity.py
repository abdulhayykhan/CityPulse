"""
intensity.py – Relative audio intensity (decibel) calculation for CityPulse.

Public API
----------
raw_db(y) -> float
    Computes the root-mean-square (RMS) level of a waveform in raw decibels
    relative to full scale (dBFS), with a floor offset for numerical stability.
"""

import numpy as np


def raw_db(y: np.ndarray) -> float:
    """Compute the raw decibel (dB) level of an audio waveform.

    Parameters
    ----------
    y : np.ndarray
        Audio signal array (typically float32 or float64).

    Returns
    -------
    float
        Raw signal level in decibels: 20 * log10(rms + 1e-9).
        Calculated using float64 precision to avoid numerical underflow.

    Notes
    -----
    * rms is calculated as sqrt(mean(y^2)).
    * An epsilon of 1e-9 prevents log10(0) for completely silent clips.
    """
    rms = np.sqrt(np.mean(y.astype(np.float64) ** 2))
    return float(20 * np.log10(rms + 1e-9))

import librosa
from src.preprocessing.audio import SR, DURATION


def raw_db_unpadded(path) -> float:
    """Compute the raw decibel (dB) level of an audio file without zero-padding.

    Loads the audio file at the target sampling rate (SR) in mono, truncates the
    waveform to at most SR * DURATION samples if it exceeds the target duration,
    and calculates its RMS decibel level using raw_db().

    Parameters
    ----------
    path : str or Path
        Path to the audio file.

    Returns
    -------
    float
        Raw decibel level computed over unpadded samples up to SR * DURATION.
    """
    # Load audio at fixed sample rate in mono without zero-padding
    y, _ = librosa.load(path, sr=SR, mono=True)

    # Truncate to at most SR * DURATION samples (first 4 seconds)
    y = y[:SR * DURATION]

    # Compute RMS dB level on unpadded waveform
    return raw_db(y)
