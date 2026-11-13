"""
Guide Part 2, Step 7 - flood cross-set test with a bootstrap confidence interval

Evaluates flood vs non-flood generalization across datasets:

  Experiment A  (always)   AlleyFloodNet official TRAIN -> AlleyFloodNet official TEST
                            = in-dataset result (what the classifier can do on the same source)
  Experiment B  (always)   AlleyFloodNet-trained detector -> RDD2022 road photos
                            = external FALSE-ALARM test. RDD2022 images contain no flooding, so
                            every 'flood' prediction there is an error. Measures false positives only.
  Experiment C  (via --flood2-root)
                            Cross-dataset validation against a 2nd flood dataset such as:
                            - Image Dataset for Roadway Flooding (Mendeley Data, DOI: 10.17632/t395bwcvbw.1)
                              with images/*.jpg + labels/*.png masks (441 flooded roadway photos)
                            - or a folder containing flood/ and non_flood/ sub-folders.
                            Computes C1 (AlleyFloodNet -> Dataset2), C1b combined cross-set with
                            external RDD2022 negatives (giving non-degenerate flood_precision & recall)
                            when Dataset2 is positive-only, and C2 reverse (Dataset2 -> AlleyFloodNet)
                            whenever Dataset2 contains both classes.

Every accuracy comes with a 95% percentile bootstrap CI (2,000 resamples of the TEST images).
Do not over-claim on small or single-polarity external sets.

Run after s5 (needs the embeddings).
"""
import argparse
import os
from pathlib import Path
import warnings

os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import common as cm
import config as C
from s4_prepare_flood import flood_class_of

warnings.filterwarnings("ignore", category=FutureWarning)


def predict_via_proba(model, X):
    proba = model.predict_proba(X)
    return model.classes_[proba.argmax(axis=1)]


def proba_binary_f1(estimator, X, y):
    pred = predict_via_proba(estimator, X)
    return f1_score(y, pred, pos_label=1, zero_division=0)


def fit_detector(X, y, n_jobs=-1):
    min_cls = int(min((y == 0).sum(), (y == 1).sum()))
    n_splits = max(2, min(3, min_cls))
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("svc", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=C.SEED)),
    ])
    grid = {"svc__C": [0.1, 1, 10, 100], "svc__gamma": ["scale", 0.001, 0.0001]}
    gs = GridSearchCV(pipe, grid, scoring=proba_binary_f1, cv=n_splits, refit=True, n_jobs=n_jobs)
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_


def _find_mask(img_path, labels_dir):
    stem = img_path.stem
    cands = [stem]
    if stem.lower().startswith("image_"):
        cands.append("label_" + stem[6:])
    elif stem.lower().startswith("image"):
        cands.append("label" + stem[5:])
    for cs in cands:
        for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
            m = labels_dir / f"{cs}{ext}"
            if m.exists():
                return m
    return None


def load_flood2_dataset(root):
    """
    Load --flood2-root supporting:
      1) Mendeley Roadway Flooding layout (images/ + labels/ segmentation masks, or Dataset/{images,labels})
      2) flood/ and non_flood/ (or flooded/ and non_flooded/) sub-folders.
    Returns (paths, y2, layout_label).
    """
    root = Path(root)
    if not root.exists():
        raise SystemExit(f"[error] --flood2-root folder not found: {root}")

    for base in (root, root / "Dataset"):
        imgs_dir, labs_dir = base / "images", base / "labels"
        if imgs_dir.is_dir() and labs_dir.is_dir():
            items, missing = [], []
            for p in cm.list_images(imgs_dir):
                m = _find_mask(p, labs_dir)
                if m is None:
                    missing.append(p.name)
                    continue
                with Image.open(m) as mask_im:
                    is_flood = bool((np.asarray(mask_im) > 0).any())
                items.append((p, 1 if is_flood else 0))
            if missing:
                raise SystemExit(
                    f"[error] {len(missing)} image(s) under {imgs_dir} have no matching mask under {labs_dir}: {missing[:5]}"
                )
            if not items:
                raise SystemExit(f"[error] no images found under {imgs_dir}")
            paths = [str(p) for p, _ in items]
            y2 = np.array([v for _, v in items], dtype=int)
            return paths, y2, "Mendeley images/labels masks"

    items = []
    for p in cm.list_images(root):
        parts_lower = {part.lower() for part in p.relative_to(root).parts[:-1]}
        if parts_lower & {"label", "labels", "mask", "masks"}:
            continue
        k = flood_class_of(p)
        if k is None:
            for part in reversed(p.relative_to(root).parts[:-1]):
                name = part.lower().replace("-", "_").replace(" ", "_")
                if "non" in name or "no_flood" in name or name in ("dry", "normal", "not_flooded"):
                    k = "non_flood"
                    break
                if "flood" in name:
                    k = "flood"
                    break
        if k:
            items.append((p, 1 if k == "flood" else 0))
    if not items:
        raise SystemExit(
            "[error] no images found in Mendeley images/labels layout or flood/non_flood folders under --flood2-root: "
            + str(root)
        )
    paths = [str(p) for p, _ in items]
    y2 = np.array([v for _, v in items], dtype=int)
    return paths, y2, "flood/non_flood folders"


def evaluate(name, train_set, test_set, model, X, y, note="", has_pos=None):
    pred = predict_via_proba(model, X)
    correct = (pred == y).astype(float)
    acc, lo, hi = cm.bootstrap_ci(correct)
    pos_present = bool((y == 1).sum() > 0) if has_pos is None else bool(has_pos)
    neg_present = bool((y == 0).sum() > 0)
    prec = round(float(precision_score(y, pred, pos_label=1, zero_division=0)), 4) if (pos_present and neg_present) else np.nan
    rec = round(float(recall_score(y, pred, pos_label=1, zero_division=0)), 4) if pos_present else np.nan
    far = round(float(((pred == 1) & (y == 0)).sum() / (y == 0).sum()), 4) if neg_present else np.nan
    row = dict(
        experiment=name, train_set=train_set, test_set=test_set,
        n_test=int(len(y)), n_flood_test=int((y == 1).sum()), n_nonflood_test=int((y == 0).sum()),
        accuracy=round(acc, 4), ci95_low=round(lo, 4), ci95_high=round(hi, 4),
        flood_precision=prec, flood_recall=rec, false_alarm_rate=far, note=note,
    )
    prec_s = "n/a" if np.isnan(prec) else f"{prec:.3f}"
    rec_s = "n/a" if np.isnan(rec) else f"{rec:.3f}"
    print(f"  {name}: acc {acc:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  prec={prec_s}  rec={rec_s}  "
          f"(n={len(y)}, flood={int((y == 1).sum())}, non_flood={int((y == 0).sum())})")
    return row, confusion_matrix(y, pred, labels=[0, 1])


def _default_flood2_root():
    for cand in (C.RAW_DIR, C.RAW_DIR / "Dataset"):
        if (cand / "images").is_dir() and (cand / "labels").is_dir():
            return str(cand)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flood2-root", default=None,
                    help="second flood dataset (Mendeley images/labels or flood/non_flood folders; "
                         "auto-detects data/raw if Mendeley layout is present)")
    ap.add_argument("--skip-flood2", action="store_true", help="skip Experiment C even if data/raw has Mendeley data")
    ap.add_argument("--n-jobs", type=int, default=-1)
    a = ap.parse_args()
    cm.ensure_dirs()

    flood2_root = None if a.skip_flood2 else (a.flood2_root if a.flood2_root else _default_flood2_root())

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
    det, params = fit_detector(X_al[tr], y_al[tr], n_jobs=a.n_jobs)
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

    # ---------------- C: true cross-set validation
    if flood2_root:
        print(f"\nExperiment C - second flood dataset: {flood2_root}")
        paths, y2, layout = load_flood2_dataset(flood2_root)
        sd = torch.load(C.PROC / "backbone_mobilenet_v2.pt", map_location="cpu")
        bb = cm.build_backbone(pretrained=False, state_dict=sd)
        X2, ok = cm.embed_paths(paths, bb)
        y2 = y2[ok]
        n_pos2, n_neg2 = int((y2 == 1).sum()), int((y2 == 0).sum())
        print(f"  dataset 2 ({layout}): {len(y2)} images | flood={n_pos2}, non_flood={n_neg2}")

        X_all_al, y_all_al = X_al, y_al
        det1, p1 = fit_detector(X_all_al, y_all_al, n_jobs=a.n_jobs)
        print("  C1 detector params:", p1)
        is_mendeley = "Mendeley" in layout
        c1_name = "C1 cross-set (Mendeley roadway)" if is_mendeley else "C1 cross-set"
        c1_test_label = "Mendeley Roadway Flooding" if is_mendeley else "Dataset2(all)"
        c1_note = (
            f"train AlleyFloodNet -> test {c1_test_label} ({layout})"
            if n_neg2 > 0
            else f"positive-only dataset (n_nonflood=0, {layout}); accuracy = flood_recall"
        )
        r, c = evaluate(c1_name, "AlleyFloodNet(all)", c1_test_label, det1, X2, y2, note=c1_note)
        rows.append(r)
        cms["C1"] = c

        # When Dataset2 contains flood positives only (e.g. Mendeley roadway flooding),
        # combine with unseen RDD2022 test road images as negatives so flood_precision is non-degenerate.
        if n_neg2 == 0 and rd.sum():
            rd_neg = rd & (M.split == "test").to_numpy()
            neg_desc = "RDD2022 test"
            if rd_neg.sum() == 0:
                rd_neg = rd
                neg_desc = "RDD2022 all"
            X_comb = np.vstack([X2, X[rd_neg]])
            y_comb = np.concatenate([y2, np.zeros(int(rd_neg.sum()), dtype=int)])
            r, c = evaluate(
                "C1b cross-set (Mendeley + RDD2022 negatives)" if is_mendeley else "C1b cross-set (+ RDD2022 negatives)",
                "AlleyFloodNet(all)",
                f"{c1_test_label} + {neg_desc}",
                det1, X_comb, y_comb,
                note=f"two-class cross-dataset test: Mendeley flood (n={n_pos2}) + {neg_desc} non-flood (n={int(rd_neg.sum())})",
            )
            rows.append(r)
            cms["C1_combined"] = c

        if len(np.unique(y2)) >= 2 and min(n_pos2, n_neg2) >= 2:
            det2, p2 = fit_detector(X2, y2, n_jobs=a.n_jobs)
            print("  C2 detector params:", p2)
            r, c = evaluate("C2 cross-set (reverse)", c1_test_label, "AlleyFloodNet(all)", det2, X_all_al, y_all_al,
                            note="train Dataset2 -> test AlleyFloodNet")
            rows.append(r)
            cms["C2"] = c
        else:
            print(f"  [info] C2 reverse cross-set skipped: Dataset2 is single-class (flood={n_pos2}, non_flood={n_neg2}); "
                  "cannot fit a binary classifier on Dataset2 alone.")
    else:
        print("\nExperiment C skipped: no second flood dataset (--flood2-root).")

    res = pd.DataFrame(rows)
    res.to_csv(C.RESULTS / "flood_crossset_results.csv", index=False)

    # figure
    fig, ax = plt.subplots(figsize=(8.6, max(3.6, 0.78 * len(res) + 1.2)))
    y_pos = np.arange(len(res))[::-1]
    ax.errorbar(res.accuracy, y_pos, xerr=[res.accuracy - res.ci95_low, res.ci95_high - res.accuracy],
                fmt="o", capsize=4, color="#4C72B0")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{e}\n(n={n})" for e, n in zip(res.experiment, res.n_test)], fontsize=8)
    ax.set_xlim(0, 1.04)
    ax.set_xlabel("accuracy with 95% bootstrap CI")
    ax.set_title("Flood vs non-flood: in-dataset and cross-set accuracy (95% bootstrap CI)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(C.RESULTS / "flood_crossset_ci.png", dpi=150)
    plt.close()

    for tag, cmat in cms.items():
        pd.DataFrame(cmat, index=["true_nonflood", "true_flood"], columns=["pred_nonflood", "pred_flood"]).to_csv(
            C.RESULTS / f"flood_confusion_matrix_{tag}.csv"
        )
    print("\nRESULT TABLE")
    cols = ["experiment", "n_test", "n_flood_test", "accuracy", "ci95_low", "ci95_high",
            "flood_precision", "flood_recall", "false_alarm_rate"]
    print(res[cols].to_string(index=False))
    print("\nLimitation to state in the report: flood data is small, comes from few scenes/sources, and "
          "images from one source can share backgrounds, so in-dataset accuracy is optimistic.")
    print("DONE (Step 7). Next: python src/image/s8_package_demo.py")


if __name__ == "__main__":
    main()
