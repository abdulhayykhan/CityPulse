# Image Pipeline Constraints

*   **Data Leakage Prevention:** Train/val/test splits must group by physical location, intersection, or source camera. Random image-level splitting is prohibited to prevent spatial leakage.
*   **Preprocessing:** All images must be resized to 224x224 pixels and normalized using standard ImageNet mean and standard deviation.
*   **Feature Extraction:** The pipeline standardizes on the MobileNetV2 backbone (`backbone_mobilenet_v2.pt`).
*   **Evaluation:** Final model evaluation must include macro F1-score and confusion matrices for both validation and test sets to account for class imbalances.