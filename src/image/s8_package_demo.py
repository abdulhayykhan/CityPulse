"""
Guide Part 2, Step 8 - package for Zainab

  * models/image_model.joblib  is written by s6_train_models.py
  * src/models/image.py        is the predict_image(path) interface (already in the repo)
  * this script copies 10 HELD-OUT TEST photos (every class covered) into demo_samples/images/
    and calls predict_image on each one.

Demo photos come from the TEST split only (never seen in training).
In PATCH mode the pothole/crack demo files are the WHOLE original photos (the app receives
whole photos), chosen from images that belong to test groups.

By default photos are picked at random per class and the prediction is shown honestly
(some may be wrong - that is real behaviour). For a rehearsed live demo you may add
--only-correct, but then say in the report that demo photos were chosen among correct ones.

Done when: Zainab calls predict_image on a demo photo and gets a sensible answer.
"""
import argparse
import json
import random
import shutil
import sys

import pandas as pd

import common as cm
import config as C

sys.path.insert(0, str(C.ROOT))
from src.models.image import predict_image  # noqa: E402


def quotas(classes, total=10):
    base = {"pothole": 3, "crack": 3, "normal": 2, "flood": 2}
    if "flood" not in classes:
        base = {"pothole": 4, "crack": 3, "normal": 3}
    q = {c: base.get(c, 1) for c in classes}
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-correct", action="store_true")
    ap.add_argument("--seed", type=int, default=C.SEED)
    a = ap.parse_args()
    rnd = random.Random(a.seed)

    man = pd.read_csv(C.PROC / "manifest.csv")
    test = man[man.split == "test"].copy()
    classes = sorted(test.label.unique())
    idx_path = C.PROC / "rdd_index.csv"
    idx = pd.read_csv(idx_path).set_index("image_id") if idx_path.exists() else None

    if C.DEMO_DIR.exists():
        for f in C.DEMO_DIR.glob("*"):
            if f.is_file():
                f.unlink()
    C.DEMO_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for cls, q in quotas(classes).items():
        sub = test[test.label == cls]
        cands = []
        for r in sub.itertuples():
            use_whole = (r.mode == "patch" and r.source == "rdd2022" and cls in ("pothole", "crack"))
            if use_whole:
                if idx is None or r.group not in idx.index or idx.loc[r.group, "image_label"] != cls:
                    continue
                cands.append((r.group, r.origin_path))
            else:
                cands.append((r.group, str(C.DATASET_DIR / r.rel_path)))
        seen, uniq = set(), []
        for g, p in cands:
            if g not in seen:
                seen.add(g)
                uniq.append((g, p))
        rnd.shuffle(uniq)
        taken = 0
        for g, p in uniq[:80]:
            if taken >= q:
                break
            out = predict_image(p)
            if a.only_correct and out["type"] != cls:
                continue
            taken += 1
            ext = p.rsplit(".", 1)[-1].lower()
            dst = C.DEMO_DIR / f"{cls}_{taken:02d}.{ext}"
            shutil.copy2(p, dst)
            rows.append(dict(file=dst.name, true_label=cls, predicted=out["type"], confidence=out["confidence"],
                             correct=out["type"] == cls, source=man[man.group == g].source.iloc[0], origin=p))
        if taken < q:
            print(f"[warn] only {taken}/{q} demo photos found for class '{cls}'")

    d = pd.DataFrame(rows)
    d.to_csv(C.DEMO_DIR / "demo_manifest.csv", index=False)
    print("\nDEMO PHOTOS (held-out test split):")
    print(d[["file", "true_label", "predicted", "confidence", "correct"]].to_string(index=False))
    print(f"\n{len(d)} files in {C.DEMO_DIR}  | correct: {int(d.correct.sum())}/{len(d)}")
    # ---- patch mode: the app sends WHOLE photos, so measure that explicitly on held-out images
    mode_file = C.PROC / "dataset_mode.json"
    mode = json.loads(mode_file.read_text()).get("mode") if mode_file.exists() else "image"
    if mode == "patch" and idx is not None:
        gs = set(test[test.source == "rdd2022"].group)
        whole = idx[idx.index.isin(gs) & idx.image_label.isin(["pothole", "crack", "normal"])]
        whole = whole.sample(n=min(len(whole), 150), random_state=a.seed)
        recs = []
        for iid, r in whole.iterrows():
            recs.append(dict(image_id=iid, true_label=r.image_label, predicted=predict_image(r.image_path)["type"]))
        w = pd.DataFrame(recs)
        w["correct"] = w.true_label == w.predicted
        tab = w.groupby("true_label").correct.agg(["mean", "count"]).rename(columns={"mean": "accuracy", "count": "n_photos"})
        tab.loc["ALL"] = [w.correct.mean(), len(w)]
        tab.round(4).to_csv(C.RESULTS / "whole_photo_check_patch_mode.csv")
        print("\n[PATCH MODE] whole-photo check on held-out test photos (model was trained on crops):")
        print(tab.round(3).to_string())
        print("-> report this honestly: patch-trained accuracy can differ from whole-photo accuracy.")

    print("\nTell Zainab:")
    print("  from src.models.image import predict_image")
    print("  predict_image('demo_samples/images/pothole_01.jpg')  ->  {'type': ..., 'confidence': ...}")
    print("DONE (Step 8).")


if __name__ == "__main__":
    main()
