"""
Guide Part 2, Step 7 - flood cross-set test with a bootstrap confidence interval

The guide wants: train the flood vs non-flood decision on ONE dataset, test on the OTHER.
You currently have ONE flood dataset (AlleyFloodNet), so this script reports honestly:

  Experiment A  (always)   AlleyFloodNet official TRAIN -> AlleyFloodNet official TEST
                            = in-dataset result (what the classifier can do on the same source)
  Experiment B  (always)   AlleyFloodNet-trained detector -> RDD2022 road photos
                            = external FALSE-ALARM test. RDD2022 images contain no flooding, so
                            every 'flood' prediction there is an error. This is a real cross-dataset
                            test, but it measures false positives only - it cannot measure recall.
  Experiment C  (optional) --flood2-root <folder with flood/ and non_flood/ sub-folders>
                            the true two-way cross-set test (the second dataset from the guide,
                            e.g. 'Floodwater on roadways', arXiv 1909.00125):
                            C1: AlleyFloodNet -> dataset 2     C2: dataset 2 -> AlleyFloodNet

Every accuracy comes with a 95% percentile bootstrap CI (2,000 resamples of the TEST images).
Small test sets give wide intervals - that is the point. Do not over-claim.

Run after s5 (needs the embeddings).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import common as cm
import config as C
from s4_prepare_flood import flood_class_of


def fit_detector(X, y):
    pipe = Pipeline([("sc", StandardScaler()), ("svc", SVC(kernel="rbf", gamma="scale", class_weight="balanced"))])
    gs = GridSearchCV(pipe, {"svc__C": [0.1, 1, 10]}, scoring="f1", cv=3, n_jobs=-1)
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_


def evaluate(name, train_set, test_set, model, X, y, note="", has_pos=True):
    pred = model.predict(X)
    correct = (pred == y).astype(float)
    acc, lo, hi = cm.bootstrap_ci(correct)
    row = dict(experiment=name, train_set=train_set, test_set=test_set, n_test=len(y),
               n_flood_test=int(y.sum()), accuracy=round(acc, 4), ci95_low=round(lo, 4), ci95_high=round(hi, 4),
               flood_precision=round(precision_score(y, pred, zero_division=0), 4) if has_pos else np.nan,
               flood_recall=round(recall_score(y, pred, zero_division=0), 4) if has_pos else np.nan,
               false_alarm_rate=round(float(((pred == 1) & (y == 0)).sum() / max((y == 0).sum(), 1)), 4),
               note=note)
    print(f"  {name}: acc {acc:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  (n={len(y)}, flood={int(y.sum())})")
    return row, confusion_matrix(y, pred, labels=[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flood2-root", default=None, help="second flood dataset with flood/ and non_flood/ folders")
    a = ap.parse_args()
    cm.ensure_dirs()

    X, M = [], []
    for s in ("train", "val", "test"):
        X.append(np.load(C.PROC / f"X_{s}.npy"))
        M.append(pd.read_csv(C.PROC / f"meta_{s}.csv"))
    X = np.concatenate(X)
    M = pd.concat(M, ignore_index=True)

    al = M.source == "alleyfloodnet"
    if al.sum() == 0:
        raise SystemExit("[error] no AlleyFloodNet rows in the manifest (you used --skip-flood?). Nothing to test.")
    y_al = (M.loc[al, "label"] == "flood").astype(int).to_numpy()
    X_al, orig = X[al.to_numpy()], M.loc[al, "orig_split"].to_numpy()
    tr, te = orig != "test", orig == "test"
    if te.sum() == 0:
        raise SystemExit("[error] AlleyFloodNet has no official test split in the manifest")

    print(f"AlleyFloodNet: official train {tr.sum()} (flood {y_al[tr].sum()}), official test {te.sum()} (flood {y_al[te].sum()})")
    rows, cms = [], {}

    # ---------------- A: in-dataset
    print("\nExperiment A - AlleyFloodNet train -> AlleyFloodNet test")
    det, params = fit_detector(X_al[tr], y_al[tr])
    print("  detector params:", params)
    r, c = evaluate("A in-dataset", "AlleyFloodNet(train)", "AlleyFloodNet(test)", det, X_al[te], y_al[te],
                    note="same source, official split")
    rows.append(r)
    cms["A"] = c

    # ---------------- B: external negatives
    print("\nExperiment B - AlleyFloodNet detector -> RDD2022 road photos (no flooding present)")
    rd = (M.source == "rdd2022").to_numpy()
    if rd.sum():
        r, c = evaluate("B external negatives (RDD2022, all classes)", "AlleyFloodNet(train)", "RDD2022 road images",
                        det, X[rd], np.zeros(rd.sum(), dtype=int), has_pos=False,
                        note="all labels are non-flood; accuracy = 1 - false-alarm rate; recall not measurable")
        rows.append(r)
        nr = rd & (M.label == "normal").to_numpy()
        if nr.sum():
            r, _ = evaluate("B2 external negatives (RDD2022 'normal' only)", "AlleyFloodNet(train)", "RDD2022 normal",
                            det, X[nr], np.zeros(nr.sum(), dtype=int), has_pos=False, note="dry road photos only")
            rows.append(r)
    else:
        print("  (no RDD2022 rows)")

    # ---------------- C: true two-way cross-set
    if a.flood2_root:
        print(f"\nExperiment C - second flood dataset: {a.flood2_root}")
        root = Path(a.flood2_root)
        items = []
        for p in cm.list_images(root):
            k = flood_class_of(p)
            if k:
                items.append((p, 1 if k == "flood" else 0))
        if not items:
            raise SystemExit("[error] no images in flood/ or non_flood/ folders under --flood2-root")
        paths = [str(p) for p, _ in items]
        y2 = np.array([v for _, v in items])
        sd = torch.load(C.PROC / "backbone_mobilenet_v2.pt", map_location="cpu")
        bb = cm.build_backbone(pretrained=False, state_dict=sd)
        X2, ok = cm.embed_paths(paths, bb)
        y2 = y2[ok]
        print(f"  dataset 2: {len(y2)} images, flood {int(y2.sum())}")
        X_all_al, y_all_al = X_al, y_al
        det1, _ = fit_detector(X_all_al, y_all_al)
        r, c = evaluate("C1 cross-set", "AlleyFloodNet(all)", "Dataset2(all)", det1, X2, y2, note="train A -> test B")
        rows.append(r)
        cms["C1"] = c
        det2, _ = fit_detector(X2, y2)
        r, c = evaluate("C2 cross-set", "Dataset2(all)", "AlleyFloodNet(all)", det2, X_all_al, y_all_al, note="train B -> test A")
        rows.append(r)
        cms["C2"] = c
    else:
        print("\nExperiment C skipped: no second flood dataset (--flood2-root). "
              "The guide's two-way cross-set test therefore could NOT be run; report A and B only.")

    res = pd.DataFrame(rows)
    res.to_csv(C.RESULTS / "flood_crossset_results.csv", index=False)

    # figure
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(res))[::-1]
    ax.errorbar(res.accuracy, y, xerr=[res.accuracy - res.ci95_low, res.ci95_high - res.accuracy], fmt="o", capsize=4, color="#4C72B0")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{e}\n(n={n})" for e, n in zip(res.experiment, res.n_test)], fontsize=7)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("accuracy with 95% bootstrap CI")
    ax.set_title("Flood vs non-flood: in-dataset and cross-set accuracy")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(C.RESULTS / "flood_crossset_ci.png", dpi=150)
    plt.close()

    cmA = cms["A"]
    pd.DataFrame(cmA, index=["true_nonflood", "true_flood"], columns=["pred_nonflood", "pred_flood"]).to_csv(
        C.RESULTS / "flood_confusion_matrix_A.csv")
    print("\nRESULT TABLE")
    print(res[["experiment", "n_test", "accuracy", "ci95_low", "ci95_high", "flood_recall", "false_alarm_rate"]].to_string(index=False))
    print("\nLimitation to state in the report: flood data is small, comes from few scenes/sources, and "
          "images from one source can share backgrounds, so in-dataset accuracy is optimistic.")
    print("DONE (Step 7). Next: python src/image/s8_package_demo.py")


if __name__ == "__main__":
    main()
