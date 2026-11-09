"""
Guide Part 2, Step 3 - build the classification dataset from RDD2022 (India)

Image-level labels (preferred, because the app receives a whole photo):
    pothole : image has at least one D40
    crack   : image has cracks (D00/D10/D20) but no D40
    normal  : image has no annotations
If fewer than MIN_NORMAL_IMAGES (300) images are 'normal', the script switches
to PATCH mode: it crops pothole/crack boxes (skipping boxes < 32 px) and cuts
random normal patches that overlap no box.

Either way:
  * each class is capped at ~1,500 samples
  * the split is 70/15/15 BY IMAGE ID (GroupShuffleSplit) - crops of one photo
    never land in both train and test
  * your Kaggle train/val/test folders are POOLED first and re-split, as the
    guide asks. Images with no label file in a split that has no labels at all
    (RDD2022's original test set is unlabeled) are excluded - never treated as 'normal'.

Output: data/processed/image/dataset/{train,val,test}/{pothole,crack,normal}/...
        data/processed/image/manifest_rdd.csv
        results/image/dataset_counts_rdd.csv
"""
import argparse
import json
import random
import shutil

import pandas as pd
from PIL import Image

import common as cm
import config as C


def overlaps(box, boxes):
    x1, y1, x2, y2 = box
    for _, (a1, b1, a2, b2) in boxes:
        if min(x2, a2) > max(x1, a1) and min(y2, b2) > max(y1, b1):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rdd-root", default=str(C.RDD_ROOT))
    ap.add_argument("--country", default=C.COUNTRY)
    ap.add_argument("--yolo-classes", default=None)
    ap.add_argument("--missing-labels", choices=["auto", "empty", "skip"], default="auto")
    ap.add_argument("--mode", choices=["auto", "image", "patch"], default="auto")
    ap.add_argument("--cap", type=int, default=C.CAP_PER_CLASS)
    ap.add_argument("--seed", type=int, default=C.SEED)
    a = ap.parse_args()

    cm.ensure_dirs()
    random.seed(a.seed)
    yc = {i: n.strip() for i, n in enumerate(a.yolo_classes.split(","))} if a.yolo_classes else None
    df, class_names, _ = cm.build_rdd_index(a.rdd_root, a.country, yc, a.missing_labels)
    if df.empty:
        raise SystemExit("[error] no images found - run s1_s2_inspect_rdd.py first and fix the path/country")

    n_unl = int((df.label_status == "unlabeled").sum())
    n_oth = int((df.image_label == "other_only").sum())
    df = df[(df.label_status != "unlabeled") & (df.image_label != "other_only")].reset_index(drop=True)
    vc = df.image_label.value_counts()
    print(f"usable labeled images: {len(df)}  (excluded: {n_unl} unlabeled, {n_oth} with only non-D00/10/20/40 classes)")
    print("image-level counts:", vc.to_dict())

    n_norm = int(vc.get("normal", 0))
    mode = a.mode if a.mode != "auto" else ("image" if n_norm >= C.MIN_NORMAL_IMAGES else "patch")
    print(f"\n>>> MODE = {mode.upper()}  (normal images = {n_norm}, threshold = {C.MIN_NORMAL_IMAGES})")

    # fresh output folder
    if C.DATASET_DIR.exists():
        shutil.rmtree(C.DATASET_DIR)
    C.DATASET_DIR.mkdir(parents=True)

    records = []   # one dict per final sample

    # ------------------------------------------------------------------ IMAGE mode
    if mode == "image":
        pool = df.rename(columns={"image_label": "label", "image_id": "group"})[
            ["group", "image_path", "label", "split"]].copy()
        pool = pool.rename(columns={"image_path": "origin_path", "split": "orig_split"})
        capped = []
        for lab, g in pool.groupby("label"):
            capped.append(g.sample(n=min(len(g), a.cap), random_state=a.seed))
        pool = pd.concat(capped).reset_index(drop=True)
        pool["split"] = cm.best_group_split(pool, "label", "group", seed=a.seed)
        for r in pool.itertuples():
            ext = r.origin_path.rsplit(".", 1)[-1].lower()
            rel = f"{r.split}/{r.label}/{r.group}.{ext}"
            dst = C.DATASET_DIR / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(r.origin_path, dst)
            records.append(dict(rel_path=rel, split=r.split, label=r.label, source="rdd2022",
                                group=r.group, origin_path=r.origin_path, orig_split=r.orig_split,
                                box="", mode="image"))

    # ------------------------------------------------------------------ PATCH mode
    else:
        cand, boxes_all, sizes = [], {}, {}
        for r in df.itertuples():
            with Image.open(r.image_path) as im:
                W, H = im.size
            objs = cm.parse_label_file(r.label_path, r.kind, class_names) if r.label_status == "labeled" else []
            boxes = cm.boxes_in_pixels(objs, r.kind, W, H)
            boxes_all[r.image_id], sizes[r.image_id] = boxes, (W, H)
            for cls, (x1, y1, x2, y2) in boxes:
                if (x2 - x1) < C.MIN_BOX_PX or (y2 - y1) < C.MIN_BOX_PX:
                    continue
                lab = "pothole" if cls == C.POTHOLE_CLASS else "crack" if cls in C.CRACK_CLASSES else None
                if lab:
                    cand.append(dict(group=r.image_id, label=lab, origin_path=r.image_path,
                                     orig_split=r.split, box=(x1, y1, x2, y2)))
        cand = pd.DataFrame(cand)
        print("candidate boxes >=32px:", cand.label.value_counts().to_dict())
        parts = [g.sample(n=min(len(g), a.cap), random_state=a.seed) for _, g in cand.groupby("label")]

        # random normal patches that overlap no annotated box
        normal, ids = [], list(df.image_id)
        by_id = df.set_index("image_id")
        for _pass in range(3):
            random.shuffle(ids)
            for iid in ids:
                if len(normal) >= a.cap:
                    break
                W, H = sizes[iid]
                for _try in range(10):
                    s = random.randint(96, max(97, min(W, H, 320)))
                    s = min(s, W, H)
                    x1, y1 = random.randint(0, W - s), random.randint(0, H - s)
                    box = (x1, y1, x1 + s, y1 + s)
                    if not overlaps(box, boxes_all[iid]):
                        normal.append(dict(group=iid, label="normal", origin_path=by_id.loc[iid, "image_path"],
                                           orig_split=by_id.loc[iid, "split"], box=box))
                        break
            if len(normal) >= a.cap:
                break
        parts.append(pd.DataFrame(normal))
        pool = pd.concat(parts).reset_index(drop=True)
        pool["split"] = cm.best_group_split(pool, "label", "group", seed=a.seed)

        k = {}
        for r in pool.itertuples():
            n = k.get(r.group, 0)
            k[r.group] = n + 1
            rel = f"{r.split}/{r.label}/{r.group}_{n}.jpg"
            dst = C.DATASET_DIR / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(r.origin_path) as im:
                im.convert("RGB").crop(tuple(int(v) for v in r.box)).save(dst, quality=95)
            records.append(dict(rel_path=rel, split=r.split, label=r.label, source="rdd2022",
                                group=r.group, origin_path=r.origin_path, orig_split=r.orig_split,
                                box=json.dumps([round(v, 1) for v in r.box]), mode="patch"))

    man = pd.DataFrame(records)
    man.to_csv(C.PROC / "manifest_rdd.csv", index=False)
    (C.PROC / "dataset_mode.json").write_text(json.dumps({"mode": mode, "normal_images_found": n_norm,
                                                          "cap": a.cap, "seed": a.seed}, indent=2))

    counts = pd.crosstab(man["label"], man["split"]).reindex(columns=["train", "val", "test"]).fillna(0).astype(int)
    counts["total"] = counts.sum(axis=1)
    counts.loc["TOTAL"] = counts.sum()
    counts.to_csv(C.RESULTS / "dataset_counts_rdd.csv")

    print("\nClass counts per split (RDD2022 only):")
    print(counts.to_string())
    leak = set(man[man.split == "train"].group) & set(man[man.split == "test"].group)
    leak |= set(man[man.split == "train"].group) & set(man[man.split == "val"].group)
    leak |= set(man[man.split == "val"].group) & set(man[man.split == "test"].group)
    print(f"\nimage ids shared between splits (must be 0): {len(leak)}")
    assert not leak, "LEAKAGE: an image id appears in two splits"
    print(f"\nsaved dataset to {C.DATASET_DIR}")
    print("DONE (Step 3). Next: python src/image/s4_prepare_flood.py")


if __name__ == "__main__":
    main()
