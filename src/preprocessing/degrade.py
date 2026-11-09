"""
degrade.py – Acoustic degradation pipeline for CityPulse robustness testing.

Public API
----------
degrade(y, sr, rng) -> np.ndarray
    Applies realistic transmission degradation to an audio waveform:
    1. 4th-order Butterworth low-pass filter at 4,000 Hz.
    2. Additive white Gaussian noise scaled to a random SNR in [10, 20] dB.
    3. Mild clipping to 90% of post-noise peak amplitude.
degrade_file(path, seed) -> np.ndarray
    Loads a clip using load_fixed() and degrades it with a deterministic seed.
degrade_intermediates(y, sr, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]
    Helper that returns intermediate signals: (y_filtered, y_noisy, y_clipped).
"""

from pathlib import Path
from typing import Tuple, Union
import numpy as np
import scipy.signal as signal
from src.preprocessing.audio import SR, load_fixed


def degrade_intermediates(
    y: np.ndarray, sr: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply degradation pipeline and return intermediate signals at each stage.

    Parameters
    ----------
    y : np.ndarray
        Mono audio signal.
    sr : int
        Sample rate in Hz.
    rng : np.random.Generator
        NumPy random generator instance for reproducible stochastic operations.

    Returns
    -------
    y_filt : np.ndarray
        Signal after 4th-order 4 kHz low-pass Butterworth filtering.
    y_noisy : np.ndarray
        Signal after additive white Gaussian noise with uniform SNR in [10, 20] dB.
    y_clipped : np.ndarray, dtype float32
        Signal after mild clipping to 0.9 * max abs of y_noisy.
    """
    # 1. Low-pass filter: 4th-order Butterworth at 4,000 Hz using second-order sections (SOS)
    sos = signal.butter(4, 4000.0, btype="low", fs=sr, output="sos")
    y_filt = signal.sosfilt(sos, y)

    # 2. Additive white Gaussian noise scaled so SNR vs filtered signal is in [10, 20] dB
    sig_power = float(np.mean(y_filt ** 2))
    if sig_power > 0.0:
        snr_db = float(rng.uniform(10.0, 20.0))
        noise_power = sig_power / (10.0 ** (snr_db / 10.0))
        noise_sigma = np.sqrt(noise_power)
        noise = rng.normal(0.0, noise_sigma, size=y.shape)
        y_noisy = y_filt + noise
    else:
        # If signal power is 0, skip noise as per specification
        y_noisy = y_filt.copy()

    # 3. Mild clipping: clip to 90% of the peak absolute amplitude after noise
    peak = float(np.max(np.abs(y_noisy)))
    if peak > 0.0:
        clip_level = 0.9 * peak
        y_clipped = np.clip(y_noisy, -clip_level, clip_level)
    else:
        y_clipped = y_noisy.copy()

    return y_filt.astype(np.float32), y_noisy.astype(np.float32), y_clipped.astype(np.float32)


def degrade(y: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Apply the 3-stage degradation pipeline to an audio waveform.

    Parameters
    ----------
    y : np.ndarray
        Input mono audio waveform.
    sr : int
        Audio sample rate in Hz.
    rng : np.random.Generator
        NumPy random number generator for additive noise sampling.

    Returns
    -------
    np.ndarray, shape same as y, dtype float32
        Acoustically degraded waveform.
    """
    _, _, y_clipped = degrade_intermediates(y, sr, rng)
    return y_clipped


def degrade_file(path: Union[str, Path], seed: int) -> np.ndarray:
    """Load a fixed 4 s clip from disk and apply degradation with deterministic seed.

    Parameters
    ----------
    path : str or Path
        Path to the audio file.
    seed : int
        Integer seed for np.random.default_rng.

    Returns
    -------
    np.ndarray, shape (SR * DURATION,), dtype float32
        Deterministic degraded waveform.
    """
    y = load_fixed(path)
    rng = np.random.default_rng(seed)
    return degrade(y, SR, rng)

