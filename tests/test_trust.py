"""
Unit tests for src/models/trust.py (Step 6.1d & Step 6.2 / Lab 3)
==================================================================
Tests:
- CSV-only trust label classification (`src/models/trust_label_rules.csv`), whitespace/case normalization,
  3 corrected Step 6.1d behaviors, 2 conflict error conditions, gate rules, and S1/S2/S3 sensitivities
- Strictly causal nearby_prior_24h_200m (t_prior < t_current; excludes self, simultaneous, >24h, >200m)
- Leak-free OHE build_trust_features (encoder fit on train only, handle_unknown='ignore')
- train_trust_model drops 'excluded' / 'unmatched' / -1 and enforces LogisticRegression(class_weight='balanced')
- score_trust returns probabilities in [0, 1] for dict, raw DataFrame, encoded DataFrame, and 1D/2D arrays
- Public exports in src/models/__init__.py
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.models import (
    assign_trust_labels,
    build_trust_features,
    classify_resolution_text,
    compute_nearby_prior_24h_200m,
    evaluate_by_group,
    evaluate_trust_predictions,
    load_trust_model,
    save_trust_model,
    score_trust,
    train_trust_model,
)


def test_classify_resolution_text_rules_and_corrected_behaviors():
    """Verify CSV-only matching, whitespace/case folding, and the 3 Step 6.1d behavior fixes."""
    # 1. Fixed behaviour #1: DEP didn't observe a violation ... couldn't issue a summons -> invalid
    dep_no_viol = (
        "The Department of Environmental Protection (DEP) didn't observe a violation of the "
        "NYC Air or Noise Code at the time of inspection and couldn't issue a summons. "
        "If the problem still exists, please go to nyc.gov/311."
    )
    assert classify_resolution_text(dep_no_viol) == "invalid"

    # 2. Fixed behaviour #2: observed no evidence of a criminal violation -> invalid
    nypd_no_evidence = (
        "The New York City Police Department responded to the complaint and with the "
        "information available observed no evidence of a criminal violation at that time."
    )
    assert classify_resolution_text(nypd_no_evidence) == "invalid"

    # 3. Fixed behaviour #3: unable to gain entry -> excluded (no longer invalid)
    nypd_no_entry = (
        "The New York City Police Department responded to the complaint but officers were "
        "unable to gain entry into the premises."
    )
    assert classify_resolution_text(nypd_no_entry) == "excluded"

    # Case-insensitive + collapsed whitespace matching + valid/no-match defaults
    assert classify_resolution_text("  DOT   INSPECTED  THIS COMPLAINT \n\t AND   REPAIRED THE PROBLEM. ") == "valid"
    assert classify_resolution_text("Some completely novel status note.") == "excluded"
    assert classify_resolution_text("") == "excluded"
    assert classify_resolution_text(None) == "excluded"


def test_classify_resolution_text_conflict_errors():
    """Matching rules of BOTH classes OR >1 distinct rule must raise ValueError and list matches."""
    # Conflict 1: matches rules of BOTH classes (valid + invalid)
    both_classes_txt = (
        "The agency inspected this complaint and repaired the problem, "
        "and police action was not necessary."
    )
    with pytest.raises(ValueError) as exc_both:
        classify_resolution_text(both_classes_txt)
    msg_both = str(exc_both.value)
    assert "BOTH classes" in msg_both
    assert "inspected this complaint and repaired the problem" in msg_both
    assert "police action was not necessary" in msg_both

    # Conflict 2: matches MORE THAN ONE distinct rule (same class)
    multi_same_class_txt = (
        "The agency inspected this complaint and repaired the problem "
        "and also cleaned the catch basin."
    )
    with pytest.raises(ValueError) as exc_multi:
        classify_resolution_text(multi_same_class_txt)
    msg_multi = str(exc_multi.value)
    assert "more than one distinct rule" in msg_multi
    assert "inspected this complaint and repaired the problem" in msg_multi
    assert "cleaned the catch basin" in msg_multi


def test_assign_trust_labels_gate_and_sensitivities():
    """Verify Gate (empty closed_date / empty resolution_id -> excluded) and S1/S2/S3 sensitivities."""
    no_crim_existed_txt = (
        "The New York City Police Department responded to the complaint and their investigation "
        "determined that no criminal violation existed. The condition was corrected without "
        "the need to issue a summons or effect an arrest."
    )
    unable_entry_txt = (
        "The New York City Police Department responded to the complaint but officers were "
        "unable to gain entry into the premises."
    )

    # Sensitivities on classify_resolution_text
    assert classify_resolution_text(no_crim_existed_txt) == "excluded"
    assert classify_resolution_text(no_crim_existed_txt, sensitivity="S1") == "invalid"
    assert classify_resolution_text(no_crim_existed_txt, s1=True) == "invalid"
    assert classify_resolution_text(no_crim_existed_txt, sensitivity="S2") == "valid"
    assert classify_resolution_text(no_crim_existed_txt, s2=True) == "valid"

    assert classify_resolution_text(unable_entry_txt) == "excluded"
    assert classify_resolution_text(unable_entry_txt, sensitivity="S3") == "invalid"
    assert classify_resolution_text(unable_entry_txt, s3=True) == "invalid"

    with pytest.raises(ValueError):
        classify_resolution_text(no_crim_existed_txt, s1=True, s2=True)
    with pytest.raises(ValueError):
        classify_resolution_text(no_crim_existed_txt, sensitivity="INVALID_SENS")

    lookup_df = pd.DataFrame({
        "resolution_id": ["10", "20", "30", "2", "3"],
        "text": [
            "The Department inspected this complaint and repaired the problem.",
            "The Department observed no evidence of a criminal violation.",
            "Completely unseen description.",
            no_crim_existed_txt,
            unable_entry_txt,
        ],
    })
    complaints = pd.DataFrame({
        "resolution_id": ["10", "10", "20", "30", "", "2", "3", "20"],
        "closed_date": [
            "2026-01-01T12:00:00",
            "",  # Resolved but empty closed_date -> Gate excludes when apply_gate=True
            "2026-01-01T12:00:00",
            "2026-01-01T12:00:00",
            "2026-01-01T12:00:00",  # No resolution_id -> excluded even if closed_date present
            "2026-01-01T12:00:00",
            "2026-01-01T12:00:00",
            "2026-01-02T12:00:00",  # Non-Closed status WITH non-empty closed_date keeps rule label
        ],
        "status": ["Closed", "Open", "Closed", "Closed", "Closed", "Closed", "Closed", "In Progress"],
    })

    # Post-gate base
    assert assign_trust_labels(complaints, lookup_df=lookup_df).tolist() == [
        "valid", "excluded", "invalid", "excluded", "excluded", "excluded", "excluded", "invalid"
    ]
    # Pre-gate base
    assert assign_trust_labels(complaints, lookup_df=lookup_df, apply_gate=False).tolist() == [
        "valid", "valid", "invalid", "excluded", "excluded", "excluded", "excluded", "invalid"
    ]
    # Post-gate S1, S2, S3
    assert assign_trust_labels(complaints, lookup_df=lookup_df, sensitivity="S1").tolist()[5] == "invalid"
    assert assign_trust_labels(complaints, lookup_df=lookup_df, sensitivity="S2").tolist()[5] == "valid"
    assert assign_trust_labels(complaints, lookup_df=lookup_df, sensitivity="S3").tolist()[6] == "invalid"


def test_compute_nearby_prior_strictly_earlier():
    """Verify strict t_cur - 24h <= t_prior < t_cur within 200m Euclidean distance."""
    df = pd.DataFrame({
        "timestamp": [
            "2026-05-01T10:00:00",  # 0: reference origin point (future relative to #3? No, #0 is 10:00)
            "2026-05-02T10:00:00",  # 1: target at +24h00m (exact 24h boundary -> included vs #0!)
            "2026-05-02T10:00:00",  # 2: simultaneous with #1 -> neither sees the other, both see #0 & #4
            "2026-05-02T10:00:01",  # 3: +1s after #1/#2 (now outside #0's 24h window!)
            "2026-05-02T09:00:00",  # 4: 1h before #1/#2 at distance sqrt(140^2 + 140^2) = 197.99m <= 200m
            "2026-05-02T09:30:00",  # 5: 30m before #1 at distance 201m (> 200m -> excluded)
        ],
        "x_m": [0.0, 0.0, 10.0, 0.0, 140.0, 201.0],
        "y_m": [0.0, 0.0, 0.0, 0.0, 140.0, 0.0],
    })
    # Permute order to test order-invariance
    perm = [3, 5, 1, 0, 4, 2]
    counts_perm = compute_nearby_prior_24h_200m(df.iloc[perm].copy())
    # Map back to original indices
    counts = np.empty_like(counts_perm)
    counts[perm] = counts_perm

    # #0 (May 1 10:00): no prior -> 0
    assert counts[0] == 0
    # #4 (May 2 09:00, dist from #0 = 197.99m <= 200m, dt=23h): sees #0 -> 1
    assert counts[4] == 1
    # #5 (May 2 09:30, x=201): #0 is 201m away, #4 is sqrt(61^2+140^2)=152.7m away & 30m earlier -> sees #4 only -> 1
    assert counts[5] == 1
    # #1 (May 2 10:00, x=0): sees #0 (dt=24h <= 24h) and #4 (dist=197.99m, dt=1h), excludes #2 (simultaneous), #3 (future), #5 (201m) -> 2
    assert counts[1] == 2
    # #2 (May 2 10:00, x=10): sees #0 (10m, 24h), #4 (dist=sqrt(130^2+140^2)=191.05m, 1h), #5 (191m, 30m) -> 3; excludes #1 (simultaneous)
    assert counts[2] == 3
    # #3 (May 2 10:00:01, x=0): #0 is 24h00m01s ago (excluded!), sees #1, #2, #4 -> 3; excludes #5 (201m)
    assert counts[3] == 3


def test_build_trust_features_no_leakage_and_unknown_categories():
    """Fit OHE strictly on train; unknown test categories become all-zero without crashing."""
    train_df = pd.DataFrame({
        "complaint_type": ["Noise - Residential", "Noise - Commercial", "Street Condition"],
        "descriptor_group": ["banging", "music_party", "pothole"],
        "channel": ["ONLINE", "PHONE", "MOBILE"],
        "location_type": ["Residential Building/House", "Club/Bar/Restaurant", "Street/Sidewalk"],
        "hour": [22, 23, 9],
        "weekday": [5, 5, 1],
        "x_m": [100.0, 110.0, 5000.0],
        "y_m": [200.0, 210.0, 5000.0],
        "timestamp": ["2026-01-03T22:00:00", "2026-01-03T23:00:00", "2026-01-06T09:00:00"],
    })
    test_df = pd.DataFrame({
        "complaint_type": ["Brand New Type"],
        "descriptor_group": ["unseen_desc"],
        "channel": ["ONLINE"],
        "location_type": ["Park/Playground"],
        "hour": [15],
        "weekday": [3],
        "nearby_prior_24h_200m": [4],
    })

    X_tr, enc = build_trust_features(train_df, return_encoder=True)
    X_te = build_trust_features(test_df, encoder=enc)

    assert list(X_tr.columns) == list(X_te.columns)
    assert "nearby_prior_24h_200m" in X_tr.columns
    assert X_tr["nearby_prior_24h_200m"].tolist() == [0.0, 1.0, 0.0]
    assert X_te["nearby_prior_24h_200m"].iloc[0] == 4.0
    # Unseen complaint_type / descriptor / location_type / hour / weekday -> only channel_ONLINE == 1.0
    assert X_te.drop(columns=["nearby_prior_24h_200m"]).to_numpy().sum() == 1.0


def test_train_trust_model_drops_excluded_and_balanced():
    """Verify excluded/unmatched/-1 targets are dropped and class_weight='balanced' is used."""
    X = pd.DataFrame({
        "f1": [1.0, 1.2, 0.0, -1.0, -1.2, 100.0, -100.0],
        "f2": [0.5, 0.4, 0.0, -0.5, -0.4, 50.0, -50.0],
    })
    # Include both string and excluded/unmatched labels
    y_mixed = ["valid", "valid", "invalid", "invalid", "invalid", "excluded", "unmatched"]
    clf = train_trust_model(X, y_mixed)
    assert isinstance(clf, LogisticRegression)
    assert clf.class_weight == "balanced"
    assert list(clf.classes_) == [0, 1]

    # Single remaining class after dropping excluded raises ValueError
    with pytest.raises(ValueError):
        train_trust_model(X.iloc[:3], ["valid", "excluded", -1])


def test_score_trust_formats_and_persistence():
    """Verify save/load and score_trust return probabilities in [0, 1] across input types."""
    rng = np.random.default_rng(42)
    n = 60
    df = pd.DataFrame({
        "complaint_type": ["Noise"] * 30 + ["Noise - Residential"] * 30,
        "descriptor_group": ["construction"] * 30 + ["banging"] * 30,
        "channel": rng.choice(["ONLINE", "MOBILE", "PHONE"], size=n),
        "location_type": ["Street/Sidewalk"] * 30 + ["Residential Building/House"] * 30,
        "hour": rng.integers(0, 24, size=n),
        "weekday": rng.integers(0, 7, size=n),
        "nearby_prior_24h_200m": rng.integers(0, 5, size=n),
    })
    y = ["valid"] * 25 + ["excluded"] * 5 + ["invalid"] * 25 + ["excluded"] * 5

    X, enc = build_trust_features(df, return_encoder=True)
    clf = train_trust_model(X, y)

    with tempfile.TemporaryDirectory() as tmpdir:
        m_path = Path(tmpdir) / "trust_test.joblib"
        save_trust_model(clf, enc, list(X.columns), model_path=m_path)
        bundle = load_trust_model(model_path=m_path)
        assert bundle["feature_names"] == list(X.columns)

        # Raw dict -> scalar float in [0, 1]
        p_valid_dict = score_trust(df.iloc[0].to_dict(), model_path=m_path)
        p_invalid_dict = score_trust(df.iloc[30].to_dict(), model_path=m_path)
        assert isinstance(p_valid_dict, float)
        assert 0.0 <= p_valid_dict <= 1.0
        assert 0.0 <= p_invalid_dict <= 1.0
        assert p_valid_dict > p_invalid_dict

        # Raw DataFrame -> 1D ndarray
        probs_raw_df = score_trust(df, model_path=m_path)
        assert isinstance(probs_raw_df, np.ndarray)
        assert probs_raw_df.shape == (n,)
        assert np.all((probs_raw_df >= 0.0) & (probs_raw_df <= 1.0))

        # Encoded DataFrame & 2D array must agree closely
        probs_enc_df = score_trust(X, model_path=m_path)
        probs_enc_arr = score_trust(X.to_numpy(), model_path=m_path)
        np.testing.assert_allclose(probs_raw_df, probs_enc_df, atol=1e-12)
        np.testing.assert_allclose(probs_raw_df, probs_enc_arr, atol=1e-12)


def test_evaluate_helpers():
    """Verify evaluate_trust_predictions and evaluate_by_group metrics on synthetic predictions."""
    eval_df = pd.DataFrame({
        "group": ["noise", "noise", "noise", "flood", "flood"],
        "y": [1, 0, 0, 1, 0],
    })
    probs = np.array([0.9, 0.2, 0.1, 0.8, 0.3])

    res = evaluate_trust_predictions(eval_df["y"], probs, threshold=0.5, n_bins=2)
    assert res["roc_auc"] == 1.0
    assert res["pr_auc"] == 1.0
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["brier"] < 0.05

    grp_df = evaluate_by_group(eval_df, probs)
    assert set(grp_df["group"]) == {"ALL", "flood", "noise"}
    assert (grp_df["roc_auc"] == 1.0).all()

