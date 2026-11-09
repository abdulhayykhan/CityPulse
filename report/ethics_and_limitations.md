# Ethics and Limitations

CityPulse relies exclusively on public municipal records and open meteorological feeds, requiring clear documentation of label semantics, reporting biases, model failure modes, and privacy boundaries before deployment.

## 1. Proxy Labels vs. Ground-Truth Hazards

The trust model (`score_trust`) predicts documented municipal agency action—such as an NYPD summons/arrest or DEP inspection warning—from 311 resolution outcomes, not ground-truth physical hazards. Because 46.17% of the 236,507 records are excluded (unresolved, referred, or ambiguous dispositions), discrimination mirrors enforcement workflows: strong for DEP noise (ROC AUC 0.8681) and road complaints (~0.94), modest for NYPD noise (0.5929), and near chance for `sewer_backup` (0.5188; excluded under rule R1). Prior-shifted scores reflect empirical resolution rates and should inform triage rather than automate dismissal of resident reports.

## 2. Dataset Bias and Zone Normalization

Voluntary 311 reporting skews toward civically active neighborhoods and noise complaints (89.57% of clean rows). Rather than clustering on raw volumes, CityPulse fits eight operational zones (`zones.json`) across 60,729 distinct locations using `capped_balanced` weights and characterizes zone profiles through normalized `lift_by_group`, preventing high-volume reporting blocks from masking localized hazards—though under-reporting communities remain underrepresented.

## 3. Regression Limits in Extreme Events

Daily flood forecasting (`predict_flood_risk`) relies on daily Open-Meteo precipitation and 1–3 day lags, missing sub-daily cloudburst intensity and drainage-saturation nonlinearities. Consequently, linear regression severely underpredicts extreme storms: on test days exceeding 50 mm (2021–2025), flood complaints averaged 188.78 versus 76.29 predicted (970 actual vs. 64.09 predicted on 2023-09-29). Outputs should trigger categorical risk bands, not exact dispatch quotas. Likewise, `noise_model_v2` adds `hour × weekday` interactions to capture late-night weekend surges missed by additive baselines.

## 4. Clustering Leakage and Crowded-Block Memorization

Five hyper-dense noise addresses—led by `1869 83 STREET` (6,868 training rows)—skewed the `nearby_prior_24h_200m` feature negative (`-0.0121`) under naive training. Excluding these top-5 noise locations from training (`M2`) prevented "crowded block" memorization and recovered the expected positive spatial-density coefficient (`+0.0448`), while rest-of-Brooklyn checks (`R2`) monitored generalization alongside strict `t_prior < t_current` causality.

## 5. Privacy and City-Agnostic Design

All training data (NYC 311, Open-Meteo, and public audio/road corpora) is public and anonymized, containing no personally identifiable reporter information. No Karachi data was collected; projection constants and baseline priors are decoupled so CityPulse remains city-agnostic and adaptable to new municipalities once local calibration data is available.

