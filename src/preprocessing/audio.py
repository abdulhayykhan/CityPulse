"""
audio.py – Low-level audio loading and feature extraction for CityPulse.

Public API
----------
load_fixed(path)       -> np.ndarray, shape (SR*DURATION,), float32
extract_features(y)    -> np.ndarray, shape (128,), float32

Constants
---------
SR       : int  – Target sample rate (22 050 Hz)
DURATION : int  – Fixed clip length in seconds (4 s → 88 200 samples)
N_MELS   : int  – Number of mel filter-bank bands (64)
"""

import numpy as np
import librosa

# ---------------------------------------------------------------------------
# Global constants – shared by every step in the pipeline
# ---------------------------------------------------------------------------
SR: int = 22_050          # Sample rate in Hz
DURATION: int = 4         # Clip length in seconds
N_MELS: int = 64          # Mel filter-bank bands


def load_fixed(path: str) -> np.ndarray:
    """Load a mono audio clip and pad/trim it to exactly SR * DURATION samples.

    Parameters
    ----------
    path : str or Path
        Absolute or relative path to the audio file.

    Returns
    -------
    y : np.ndarray, shape (SR * DURATION,), dtype float32
        Mono waveform resampled to SR and fixed to exactly DURATION seconds.

    Notes
    -----
    * librosa.load always returns float32.
    * librosa.util.fix_length pads with zeros if the clip is shorter than
      SR * DURATION, and silently truncates if it is longer.
    """
    # Load at target SR, convert to mono if the source is stereo
    y, _ = librosa.load(path, sr=SR, mono=True)

    # Pad (zero) or trim to exactly SR * DURATION samples
    target_length = SR * DURATION
    y = librosa.util.fix_length(y, size=target_length)

    return y  # float32, shape (88200,)


def extract_features(y: np.ndarray) -> np.ndarray:
    """Compute a 128-dimensional log-mel feature vector from a waveform.

    The vector is the concatenation of [per-band mean, per-band std] of the
    log-mel spectrogram, giving a compact, fixed-length representation that
    is agnostic to clip length (provided the input is already fixed-length).

    Parameters
    ----------
    y : np.ndarray, shape (SR * DURATION,), dtype float32
        Mono waveform at SR Hz, as returned by load_fixed().

    Returns
    -------
    features : np.ndarray, shape (128,), dtype float32
        Concatenation of [logmel.mean(axis=1), logmel.std(axis=1)].
        The first 64 numbers are per-band temporal means;
        the last  64 numbers are per-band temporal standard deviations.

    Notes
    -----
    * ref=np.max normalises the dB scale so the loudest frame is 0 dB;
      this makes the representation partially amplitude-invariant.
    * The output is explicitly cast to float32 to guarantee dtype consistency
      even on platforms where numpy defaults to float64 for std().
    """
    # Compute mel-spectrogram (power spectrum projected onto mel scale)
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS)

    # Convert to log scale in dB; ref=np.max → loudest frame = 0 dB
    logmel = librosa.power_to_db(mel, ref=np.max)   # shape: (N_MELS, T)

    # Summarise each mel band over time: mean and std → 64 + 64 = 128 values
    band_mean = logmel.mean(axis=1)   # shape: (N_MELS,)
    band_std  = logmel.std(axis=1)    # shape: (N_MELS,)

    features = np.concatenate([band_mean, band_std])  # shape: (128,)

    return features.astype(np.float32)

