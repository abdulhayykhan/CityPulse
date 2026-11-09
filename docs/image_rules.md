# Image Pipeline Rules

* Always use `GroupShuffleSplit` on `image_id`.
* Never use absolute paths in output manifests.
* Always evaluate cross-dataset metrics using bootstrap confidence intervals.
* Models must be saved using `scikit-learn==1.9.1`.

