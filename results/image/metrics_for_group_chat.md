**CityPulse - Image model metrics (Bushra)**
Features: MobileNetV2 embeddings (1280-d). Split by image id 70/15/15. Dataset mode: image.

Test set:
| model | n | accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 |
|---|---|---|---|---|---|---|
| SVM | 871 | 0.719 | 0.730 | 0.722 | 0.726 | 0.717 |
| RandomForest | 871 | 0.644 | 0.644 | 0.672 | 0.655 | 0.642 |

Packaged model (best validation macro-F1): **SVM** -> models/image_model.joblib
Interface: `predict_image(path) -> {type, confidence}`

Flood (AlleyFloodNet in-dataset): accuracy 0.924, 95% CI [0.889, 0.954], n=262
