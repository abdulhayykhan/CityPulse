"""
Guide Part 2, Step 9 (OPTIONAL - only if you are ahead of schedule)

"Try horizontal flips and brightness changes on the training images and check
whether the F1 score improves."

Method: embed an extra copy of every TRAIN image after (a) horizontal flip and
(b) a random brightness change (x0.7 or x1.3), add them to the SVM training set,
re-fit the SVM with the SAME hyper-parameters found in Step 6, and compare macro-F1
on the untouched validation and test sets. Only training data is augmented.

Output: results/image/augmentation_experiment.csv
Interpret small differences (<~1 point of F1) as noise; say so in the report.
"""
import json
import random

import numpy as np
import pandas as pd
import torch
from PIL import ImageEnhance, ImageOps
from sklearn.metrics import f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import common as cm
import config as C


def main():
    cm.ensure_dirs()
    prm = json.loads((C.RESULTS / "best_params.json").read_text())["svm"]
    Xtr, ytr = np.load(C.PROC / "X_train.npy"), np.load(C.PROC / "y_train.npy")
    mtr = pd.read_csv(C.PROC / "meta_train.csv")
    Xva, yva = np.load(C.PROC / "X_val.npy"), np.load(C.PROC / "y_val.npy")
    Xte, yte = np.load(C.PROC / "X_test.npy"), np.load(C.PROC / "y_test.npy")
    paths = [str(C.DATASET_DIR / p) for p in mtr.rel_path]

    sd = torch.load(C.PROC / "backbone_mobilenet_v2.pt", map_location="cpu")
    bb = cm.build_backbone(pretrained=False, state_dict=sd)
    rnd = random.Random(C.SEED)

    print("embedding flipped copies ...")
    Xf, okf = cm.embed_paths(paths, bb, pil_transform=ImageOps.mirror)
    print("embedding brightness-changed copies ...")
    Xb, okb = cm.embed_paths(paths, bb, pil_transform=lambda im: ImageEnhance.Brightness(im).enhance(rnd.choice([0.7, 1.3])))

    def fit_eval(X, y):
        m = Pipeline([("sc", StandardScaler()), ("svc", SVC(kernel="rbf", class_weight="balanced",
                                                            C=prm["svc__C"], gamma=prm["svc__gamma"]))])
        m.fit(X, y)
        out = {}
        for nm, (Xe, ye) in {"val": (Xva, yva), "test": (Xte, yte)}.items():
            p = m.predict(Xe)
            out[nm] = (f1_score(ye, p, average="macro"), accuracy_score(ye, p))
        return out

    variants = {
        "baseline (no augmentation)": (Xtr, ytr),
        "+ horizontal flips": (np.vstack([Xtr, Xf]), np.concatenate([ytr, ytr[okf]])),
        "+ brightness changes": (np.vstack([Xtr, Xb]), np.concatenate([ytr, ytr[okb]])),
        "+ flips + brightness": (np.vstack([Xtr, Xf, Xb]), np.concatenate([ytr, ytr[okf], ytr[okb]])),
    }
    rows = []
    for name, (X, y) in variants.items():
        r = fit_eval(X, y)
        rows.append(dict(variant=name, n_train=len(y), val_macro_f1=round(r["val"][0], 4), test_macro_f1=round(r["test"][0], 4),
                         val_acc=round(r["val"][1], 4), test_acc=round(r["test"][1], 4)))
    df = pd.DataFrame(rows)
    base = df.iloc[0]
    df["test_f1_delta_vs_baseline"] = (df.test_macro_f1 - base.test_macro_f1).round(4)
    df.to_csv(C.RESULTS / "augmentation_experiment.csv", index=False)
    print(df.to_string(index=False))
    best = df.iloc[1:].sort_values("test_macro_f1", ascending=False).iloc[0]
    verdict = "improves" if best.test_f1_delta_vs_baseline > 0.01 else "does not clearly improve"
    print(f"\nBest augmentation '{best.variant}' {verdict} test macro-F1 (delta {best.test_f1_delta_vs_baseline:+.4f}). "
          "Differences under ~0.01 are within noise.")
    print("(The packaged model is NOT changed by this experiment.)")


if __name__ == "__main__":
    main()
