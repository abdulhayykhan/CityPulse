"""
CityPulse - Regression and Trust Scoring Models
"""

from src.models.regression import (
    build_flood_features,
    rolling_origin_splits,
    chrono_split,
    fit_eval,
    baseline_mean,
    build_noise_hourly,
    add_noise_interaction_features,
    evaluate_and_train_noise_interaction,
    load_flood_model,
    load_noise_model,
    load_noise_model_v2,
    predict_flood_risk,
    predict_noise,
    predict_noise_v2,
)
from src.models.trust import (
    TRUST_CATEGORICAL_COLS,
    TRUST_NUMERIC_COLS,
    classify_resolution_text,
    assign_trust_labels,
    compute_nearby_prior_24h_200m,
    build_trust_features,
    train_trust_model,
    save_trust_model,
    load_trust_model,
    score_trust,
    evaluate_trust_predictions,
    evaluate_by_group,
)

__all__ = [
    "build_flood_features",
    "rolling_origin_splits",
    "chrono_split",
    "fit_eval",
    "baseline_mean",
    "build_noise_hourly",
    "add_noise_interaction_features",
    "evaluate_and_train_noise_interaction",
    "load_flood_model",
    "load_noise_model",
    "load_noise_model_v2",
    "predict_flood_risk",
    "predict_noise",
    "predict_noise_v2",
    "TRUST_CATEGORICAL_COLS",
    "TRUST_NUMERIC_COLS",
    "classify_resolution_text",
    "assign_trust_labels",
    "compute_nearby_prior_24h_200m",
    "build_trust_features",
    "train_trust_model",
    "save_trust_model",
    "load_trust_model",
    "score_trust",
    "evaluate_trust_predictions",
    "evaluate_by_group",
]

