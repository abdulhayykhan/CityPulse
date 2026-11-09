# Audio Pipeline Rules

* Train/val/test splits must group by original audio source (e.g., fold or `fsID`). No random patch-level splitting.
* Target sampling rate is 22,050 Hz with 4-second fixed clip lengths.
* Feature extraction uses 64-band log-Mel spectrograms.
* `relative_intensity` bounds must be computed strictly on the training folds.
* Models must be trained using augmented (degraded) data to simulate poor microphone quality.
* Final model evaluation must include a true cross-dataset test.

