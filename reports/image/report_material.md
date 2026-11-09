# Image component - results (Bushra Zafar)

## 1. Dataset counts

Mode: **image**-level samples; cap 1500 per class; split 70/15/15 by image id.

| class | train | val | test | total |
|---|---|---|---|---|
| crack | 1049 | 221 | 230 | 1500 |
| flood | 397 | 71 | 133 | 601 |
| normal | 1372 | 283 | 354 | 2009 |
| pothole | 740 | 162 | 154 | 1056 |
| TOTAL | 3558 | 737 | 871 | 5166 |

By source:

| source | label | train | val | test |
|---|---|---|---|---|
| alleyfloodnet | flood | 397 | 71 | 133 |
| alleyfloodnet | normal | 323 | 57 | 129 |
| rdd2022 | crack | 1049 | 221 | 230 |
| rdd2022 | normal | 1049 | 226 | 225 |
| rdd2022 | pothole | 740 | 162 | 154 |

## 2. SVM vs Random Forest

Test:

| model | n | accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 |
|---|---|---|---|---|---|---|
| SVM | 871 | 0.719 | 0.730 | 0.722 | 0.726 | 0.717 |
| RandomForest | 871 | 0.644 | 0.644 | 0.672 | 0.655 | 0.642 |

Validation:

| model | n | accuracy | macro_f1 |
|---|---|---|---|
| SVM | 737 | 0.669 | 0.700 |
| RandomForest | 737 | 0.635 | 0.679 |

Hyper-parameters (grouped GroupKFold(5) by image id, macro-F1): SVM {'svc__C': 10, 'svc__gamma': 'scale'}, RF {'rf__n_estimators': 200}

Per-class F1 (test):

| class | RandomForest | SVM |
|---|---|---|
| crack | 0.621 | 0.665 |
| flood | 0.878 | 0.925 |
| normal | 0.636 | 0.731 |
| pothole | 0.486 | 0.580 |

Per-class precision / recall / F1 (test):

| model | class | precision | recall | f1 | support |
|---|---|---|---|---|---|
| SVM | crack | 0.683 | 0.648 | 0.665 | 230 |
| SVM | flood | 0.918 | 0.932 | 0.925 | 133 |
| SVM | normal | 0.707 | 0.757 | 0.731 | 354 |
| SVM | pothole | 0.612 | 0.552 | 0.580 | 154 |
| RandomForest | crack | 0.599 | 0.643 | 0.621 | 230 |
| RandomForest | flood | 0.818 | 0.947 | 0.878 | 133 |
| RandomForest | normal | 0.690 | 0.590 | 0.636 | 354 |
| RandomForest | pothole | 0.467 | 0.506 | 0.486 | 154 |

Test accuracy by data source:

| model | source | n | accuracy | macro_f1 |
|---|---|---|---|---|
| SVM | alleyfloodnet | 262 | 0.924 | 0.924 |
| SVM | rdd2022 | 609 | 0.630 | 0.625 |
| RandomForest | alleyfloodnet | 262 | 0.866 | 0.865 |
| RandomForest | rdd2022 | 609 | 0.548 | 0.540 |

## 3. Confusion matrix

See `confusion_matrix_svm_test.png` and `confusion_matrix_rf_test.png` (counts and row-normalised).

## 4. Flood cross-set result

| experiment | train_set | test_set | n_test | accuracy | ci95_low | ci95_high | flood_precision | flood_recall | false_alarm_rate |
|---|---|---|---|---|---|---|---|---|---|
| A in-dataset | AlleyFloodNet(train) | AlleyFloodNet(test) | 262 | 0.924 | 0.889 | 0.954 | 0.931 | 0.917 | 0.070 |
| B external negatives (RDD2022, all classes) | AlleyFloodNet(train) | RDD2022 road images | 4056 | 0.604 | 0.589 | 0.619 | - | - | 0.396 |
| B2 external negatives (RDD2022 'normal' only) | AlleyFloodNet(train) | RDD2022 normal | 1500 | 0.581 | 0.557 | 0.606 | - | - | 0.419 |
| C1 cross-set (Mendeley roadway) | AlleyFloodNet(all) | Mendeley Roadway Flooding | 441 | 0.939 | 0.916 | 0.962 | - | 0.939 | - |
| C1b cross-set (Mendeley + RDD2022 negatives) | AlleyFloodNet(all) | Mendeley Roadway Flooding + RDD2022 test | 1050 | 0.783 | 0.757 | 0.807 | 0.673 | 0.939 | 0.330 |

95% percentile bootstrap CI over test images (2,000 resamples).

## 5. Limitations (draft paragraph)

The image classifier uses frozen ImageNet MobileNetV2 embeddings with an SVM or Random Forest; it was not fine-tuned. Road-damage labels come from RDD2022 (India subset, collected from smartphone-mounted vehicles in India), so results do not describe Karachi roads. Image-level labels were derived from bounding-box annotations (pothole if any D40 box, crack if only D00/D10/D20 boxes, normal if no annotations); this is a proxy label, and an image marked 'normal' may still contain damage nobody annotated. The flood class comes from a single, small dataset (AlleyFloodNet, 601 flood images in this experiment) and is collected in a different setting from the road-damage photos, so the classifier could partly learn dataset identity rather than flooding; the non-flooded images were added to the 'normal' class to reduce this. Flood accuracy has a wide bootstrap confidence interval and a true two-dataset cross-set test was limited by data availability. Predicted confidences are relative scores, not calibrated probabilities. Reported performance is on held-out public images, not on photos taken by citizens with their own phones.
