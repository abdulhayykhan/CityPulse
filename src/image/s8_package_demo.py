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
from pathlib import Path
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

    candidates_by_cls = {}
    any_source_exists = False
    for cls in quotas(classes):
        sub = test[test.label == cls]
        cands = []
        for r in sub.itertuples():
            use_whole = (r.mode == "patch" and r.source == "rdd2022" and cls in ("pothole", "crack"))
            if use_whole:
                if idx is None or r.group not in idx.index or idx.loc[r.group, "image_label"] != cls:
                    continue
                p_str = r.origin_path
            else:
                p_str = str(C.DATASET_DIR / r.rel_path)
            if Path(p_str).exists():
                any_source_exists = True
            cands.append((r.group, p_str))
        candidates_by_cls[cls] = cands

    manifest_out = C.RESULTS / "demo_manifest.csv"
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    if any_source_exists:
        if C.DEMO_DIR.exists():
            for f in C.DEMO_DIR.glob("*"):
                if f.is_file() and f.name != ".gitkeep":
                    f.unlink()
        C.DEMO_DIR.mkdir(parents=True, exist_ok=True)

        for cls, q in quotas(classes).items():
            seen, uniq = set(), []
            for g, p in candidates_by_cls[cls]:
                if g not in seen:
                    seen.add(g)
                    uniq.append((g, p))
            rnd.shuffle(uniq)
            taken = 0
            for g, p in uniq[:80]:
                if taken >= q:
                    break
                if not Path(p).exists():
                    continue
                out = predict_image(p)
                if a.only_correct and out["type"] != cls:
                    continue
                taken += 1
                ext = p.rsplit(".", 1)[-1].lower()
                dst = C.DEMO_DIR / f"{cls}_{taken:02d}.{ext}"
                shutil.copy2(p, dst)
                try:
                    rel_origin = Path(p).resolve().relative_to(C.ROOT.resolve()).as_posix()
                except ValueError:
                    rel_origin = Path(p).name
                rows.append(dict(file=dst.name, true_label=cls, predicted=out["type"], confidence=out["confidence"],
                                 correct=out["type"] == cls, source=man[man.group == g].source.iloc[0], origin=rel_origin))
            if taken < q:
                print(f"[warn] only {taken}/{q} demo photos found for class '{cls}'")
    else:
        existing = sorted(p for p in C.DEMO_DIR.glob("*") if p.is_file() and p.suffix.lower() in cm.IMG_EXTS)
        if not existing:
            raise SystemExit(
                f"[error] no held-out test images found on disk under {C.DATASET_DIR} or in {C.DEMO_DIR}"
            )
        prev = pd.read_csv(manifest_out).set_index("file") if manifest_out.exists() else None
        print(f"[info] {C.DATASET_DIR} not present locally; re-scoring {len(existing)} existing photos in {C.DEMO_DIR}")
        for img_path in existing:
            out = predict_image(str(img_path))
            inferred_cls = img_path.stem.rsplit("_", 1)[0]
            if prev is not None and img_path.name in prev.index:
                pr = prev.loc[img_path.name]
                true_label = str(pr["true_label"])
                src_name = str(pr["source"])
                raw_orig = str(pr["origin"]).replace("\\", "/")
                if ":" in raw_orig:
                    raw_orig = f"demo_samples/images/{img_path.name}"
            else:
                true_label = inferred_cls
                src_name = "alleyfloodnet" if inferred_cls == "flood" else "rdd2022"
                raw_orig = f"demo_samples/images/{img_path.name}"
            rows.append(dict(file=img_path.name, true_label=true_label, predicted=out["type"],
                             confidence=out["confidence"], correct=out["type"] == true_label,
                             source=src_name, origin=raw_orig))

    d = pd.DataFrame(rows)
    d.to_csv(manifest_out, index=False)
    print("\nDEMO PHOTOS (held-out test split):")
    print(d[["file", "true_label", "predicted", "confidence", "correct"]].to_string(index=False))
    print(f"\n{len(d)} files in {C.DEMO_DIR} | manifest: {manifest_out} | correct: {int(d.correct.sum())}/{len(d)}")
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
