"""
Guide Part 2, Step 6 (+ packaging of the model file used in Step 8)

  * SVM:  Pipeline(StandardScaler, SVC(rbf, class_weight='balanced', probability=True))
          GridSearchCV over C and gamma, scoring = macro-F1
  * RF :  RandomForestClassifier(class_weight='balanced'), 200 vs 300 trees (GridSearchCV)
  * Cross-validation is GROUPED by image id (GroupKFold) so patches of one photo
    never sit in both a training fold and its validation fold.
  * Models are fitted on the TRAIN split only. Validation picks the winner,
    test is reported once. The reported numbers therefore describe exactly the
    model that is saved.
  * Predictions everywhere = argmax of predict_proba (that is what predict_image
    returns), so the tables match the deployed behaviour.

Outputs (results/image/):
    metrics_summary.csv        SVM vs RF table (val + test)
    per_class_metrics.csv      precision / recall / F1 per class
    test_by_source.csv         test accuracy per data source (rdd2022 vs alleyfloodnet)
    confusion_matrix_{svm,rf}_test.png / .csv
    svm_vs_rf_per_class_f1.png
    rf_feature_importance.png
    best_params.json
models/image_model.joblib      packaged best model (self-contained, incl. backbone weights)

Done when: you have an SVM vs Random Forest table.
"""
import argparse
import json
import os
import time
import warnings

os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning")   # also silences joblib worker processes

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import common as cm
import config as C

warnings.filterwarnings("ignore", category=FutureWarning)   # sklearn>=1.9 notes SVC(probability=True) will change; guide asks for it


def predict_via_proba(model, X):
    proba = model.predict_proba(X)
    return model.classes_[proba.argmax(axis=1)], proba


def summarize(name, split, y_true, y_pred, classes):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    summ = dict(model=name, split=split, n=len(y_true),
                accuracy=accuracy_score(y_true, y_pred),
                balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
                macro_precision=p.mean(), macro_recall=r.mean(), macro_f1=f.mean(), weighted_f1=fw)
    per = [dict(model=name, split=split, **{"class": c}, precision=p[i], recall=r[i], f1=f[i], support=int(s[i]))
           for i, c in enumerate(classes)]
    return summ, per


def plot_cm(cmat, classes, title, path):
    norm = cmat / np.maximum(cmat.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(cmat, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax[0], cbar=False)
    ax[0].set_title("counts")
    sns.heatmap(norm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, xticklabels=classes, yticklabels=classes, ax=ax[1], cbar=False)
    ax[1].set_title("row-normalised (recall)")
    for x in ax:
        x.set_xlabel("predicted")
        x.set_ylabel("true")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--allow-random-weights", action="store_true", help="TEST ONLY")
    a = ap.parse_args()

    cm.ensure_dirs()
    info = json.loads((C.PROC / "embedding_info.json").read_text())
    if not info["pretrained"] and not a.allow_random_weights:
        raise SystemExit("[error] embeddings were made with RANDOM weights (--no-pretrained). Re-run s5 properly.")

    D = {}
    for s in ("train", "val", "test"):
        D[s] = (np.load(C.PROC / f"X_{s}.npy"), np.load(C.PROC / f"y_{s}.npy"), pd.read_csv(C.PROC / f"meta_{s}.csv"))
    Xtr, ytr, mtr = D["train"]
    classes = sorted(set(map(str, ytr)))
    print("classes:", classes, "| train", Xtr.shape, "| val", D["val"][0].shape, "| test", D["test"][0].shape)
    groups = mtr["group"].to_numpy()
    n_groups = len(set(groups))
    cv = GroupKFold(n_splits=min(a.cv, n_groups))
    t0 = time.time()

    # ---------------------------------------------------------------- SVM
    print("\n[SVM] grid search (grouped CV, macro-F1) ...")
    svm_pipe = Pipeline([("scaler", StandardScaler()),
                         ("svc", SVC(kernel="rbf", class_weight="balanced", probability=False, random_state=C.SEED))])
    svm_grid = {"svc__C": [0.1, 1, 10, 100], "svc__gamma": ["scale", 0.001, 0.0001]}
    gs_svm = GridSearchCV(svm_pipe, svm_grid, scoring="f1_macro", cv=cv, n_jobs=a.n_jobs, refit=False, verbose=1)
    gs_svm.fit(Xtr, ytr, groups=groups)
    best_svm_params = gs_svm.best_params_
    print("[SVM] best:", best_svm_params, "| CV macro-F1 = %.4f" % gs_svm.best_score_)
    svm = Pipeline([("scaler", StandardScaler()),
                    ("svc", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=C.SEED,
                                C=best_svm_params["svc__C"], gamma=best_svm_params["svc__gamma"]))])
    svm.fit(Xtr, ytr)          # probability=True -> final fit with Platt scaling (used by predict_image)

    # ---------------------------------------------------------------- RF
    print("\n[RF] grid search (200 vs 300 trees) ...")
    def make_rf(**kw):   # scaler first: RF is scale-invariant but sklearn trees ignore features with a tiny range
        return Pipeline([("scaler", StandardScaler()),
                         ("rf", RandomForestClassifier(class_weight="balanced", n_jobs=1, random_state=C.SEED, **kw))])
    rf_grid = {"rf__n_estimators": [200, 300]}
    gs_rf = GridSearchCV(make_rf(), rf_grid, scoring="f1_macro", cv=cv, n_jobs=a.n_jobs, refit=False, verbose=1)
    gs_rf.fit(Xtr, ytr, groups=groups)
    best_rf_params = gs_rf.best_params_
    print("[RF] best:", best_rf_params, "| CV macro-F1 = %.4f" % gs_rf.best_score_)
    rf = make_rf(n_estimators=best_rf_params["rf__n_estimators"])
    rf.set_params(rf__n_jobs=-1)
    rf.fit(Xtr, ytr)

    models = {"SVM": svm, "RandomForest": rf}
    cv_scores = {"SVM": float(gs_svm.best_score_), "RandomForest": float(gs_rf.best_score_)}

    # ---------------------------------------------------------------- evaluation
    summ_rows, per_rows, src_rows = [], [], []
    for name, mdl in models.items():
        assert list(mdl.classes_) == classes
        for split in ("val", "test"):
            X, y, meta = D[split]
            pred, _ = predict_via_proba(mdl, X)
            s, per = summarize(name, split, y, pred, classes)
            s["cv_macro_f1_train"] = cv_scores[name]
            summ_rows.append(s)
            per_rows.extend(per)
            cmat = confusion_matrix(y, pred, labels=classes)
            pd.DataFrame(cmat, index=classes, columns=classes).to_csv(C.RESULTS / f"confusion_matrix_{name.lower()}_{split}.csv")
            if split == "test":
                short = "svm" if name == "SVM" else "rf"
                plot_cm(cmat, classes, f"{name} - test set (n={len(y)})", C.RESULTS / f"confusion_matrix_{short}_test.png")
                for srcname, g in meta.assign(y=y, pred=pred).groupby("source"):
                    present = [c for c in classes if c in set(g["y"])]      # macro-F1 only over classes this source has
                    ss, _ = summarize(name, "test", g["y"], g["pred"], present)
                    src_rows.append(dict(model=name, source=srcname, n=len(g), accuracy=ss["accuracy"],
                                         macro_f1=ss["macro_f1"], classes_present=sorted(set(g["y"]))))

    summary = pd.DataFrame(summ_rows)
    per_df = pd.DataFrame(per_rows)
    summary.round(4).to_csv(C.RESULTS / "metrics_summary.csv", index=False)
    per_df.round(4).to_csv(C.RESULTS / "per_class_metrics.csv", index=False)
    pd.DataFrame(src_rows).round(4).to_csv(C.RESULTS / "test_by_source.csv", index=False)

    # per-class F1 bar chart
    t = per_df[per_df.split == "test"].pivot(index="class", columns="model", values="f1").reindex(classes)
    ax = t.plot.bar(figsize=(7, 4), color=["#DD8452", "#4C72B0"])
    ax.set_ylabel("F1 (test)")
    ax.set_ylim(0, 1)
    ax.set_title("SVM vs Random Forest - per-class F1 on test")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.savefig(C.RESULTS / "svm_vs_rf_per_class_f1.png", dpi=150)
    plt.close()

    # RF importance (abstract embedding dims - labelled honestly)
    imp = pd.Series(rf.named_steps["rf"].feature_importances_).sort_values(ascending=False).head(20)
    ax = imp.iloc[::-1].plot.barh(figsize=(6, 5), color="#55A868")
    ax.set_yticklabels([f"dim {i}" for i in imp.index[::-1]])
    ax.set_title("RF: top-20 MobileNetV2 embedding dims\n(abstract features, not human-readable)")
    plt.tight_layout()
    plt.savefig(C.RESULTS / "rf_feature_importance.png", dpi=150)
    plt.close()

    (C.RESULTS / "best_params.json").write_text(json.dumps(
        dict(svm=best_svm_params, rf=best_rf_params, cv_macro_f1_train=cv_scores,
             cv="GroupKFold(%d) by image id" % cv.n_splits, scoring="f1_macro"), indent=2, default=str))

    print("\n" + "=" * 70)
    print("SVM vs RANDOM FOREST")
    print("=" * 70)
    show = summary[["model", "split", "n", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]].round(4)
    print(show.to_string(index=False))
    print("\nper-class (test):")
    print(per_df[per_df.split == "test"].round(3).to_string(index=False))
    print("\ntest accuracy by source:")
    print(pd.DataFrame(src_rows).round(3).to_string(index=False))

    # ---------------------------------------------------------------- package
    val_f1 = summary[summary.split == "val"].set_index("model")["macro_f1"]
    best_name = val_f1.idxmax()
    best = models[best_name]
    print(f"\nWinner on VALIDATION macro-F1: {best_name}  ({val_f1.to_dict()})")

    sd = {k: v.cpu().numpy() for k, v in __import__("torch").load(C.PROC / "backbone_mobilenet_v2.pt", map_location="cpu").items()}
    test_row = summary[(summary.model == best_name) & (summary.split == "test")].iloc[0]
    bundle = dict(
        estimator=best, classes=list(best.classes_), model_name=best_name,
        backbone="mobilenet_v2", backbone_state=sd, pretrained=bool(info["pretrained"]),
        img_size=C.IMG_SIZE, mean=C.IMAGENET_MEAN, std=C.IMAGENET_STD,
        test_metrics=dict(accuracy=float(test_row.accuracy), macro_f1=float(test_row.macro_f1)),
        note="predict_image(): MobileNetV2 1280-d embedding -> estimator.predict_proba; type=argmax, confidence=max proba",
    )
    out = C.MODELS_DIR / "image_model.joblib"
    joblib.dump(bundle, out, compress=3)
    mb = out.stat().st_size / 1e6
    print(f"saved {out}  ({mb:.1f} MB)")
    if mb > 50:
        print("[warn] file > 50 MB: GitHub warns above 50 MB and blocks above 100 MB. "
              "Tell the team or switch the winner to the smaller model.")

    reload = joblib.load(out)
    Xt = D["test"][0][:5]
    print("reload check, first 5 test rows ->", list(reload["estimator"].predict(Xt)), "| truth", list(D["test"][1][:5]))
    print(f"\nDONE (Step 6) in {time.time() - t0:.0f} s. Post the table above in the group chat. Next: s7, s8.")


if __name__ == "__main__":
    main()
