"""
Guide Part 2, Step 1 + Step 2
  Step 1  list India images and their label files        (Done when: you can list them)
  Step 2  print counts per damage class and the number of label files with zero objects

Your Kaggle copy of RDD2022 has train/val/test folders, no XML and no country folders:
  * country  -> taken from the file-name prefix (India_000123.jpg)
  * labels   -> YOLO .txt (class cx cy w h, normalised). Pascal VOC .xml is also
                supported automatically if present.

Usage:
    python src/image/s1_s2_inspect_rdd.py
    python src/image/s1_s2_inspect_rdd.py --rdd-root D:/data/RDD2022 --country India
    python src/image/s1_s2_inspect_rdd.py --country all          # skip the country filter
    python src/image/s1_s2_inspect_rdd.py --yolo-classes D00,D10,D20,D40
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import common as cm
import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rdd-root", default=str(C.RDD_ROOT))
    ap.add_argument("--country", default=C.COUNTRY, help="'India' (default) or 'all'")
    ap.add_argument("--yolo-classes", default=None,
                    help="comma list in class-id order, e.g. D00,D10,D20,D40 (default: data.yaml if found, else this)")
    ap.add_argument("--missing-labels", choices=["auto", "empty", "skip"], default="auto",
                    help="what an image WITHOUT a label file means (see README)")
    a = ap.parse_args()

    cm.ensure_dirs()
    yc = {i: n.strip() for i, n in enumerate(a.yolo_classes.split(","))} if a.yolo_classes else None
    df, class_names, names_src = cm.build_rdd_index(a.rdd_root, a.country, yc, a.missing_labels)

    print("=" * 70)
    print("STEP 1 - list images and label files")
    print("=" * 70)
    all_imgs = cm.list_images(a.rdd_root)
    print(f"RDD2022 root           : {a.rdd_root}")
    print(f"images found (all)     : {len(all_imgs)}")
    if df.empty:
        print(f"\n[error] 0 images match country='{a.country}'. Example file names found:")
        for p in all_imgs[:8]:
            print("   ", p.name)
        print("Fix: check the prefix, or run with --country all")
        raise SystemExit(1)

    print(f"images for '{a.country}'    : {len(df)}")
    print(f"YOLO class-id mapping  : {class_names}   (source: {names_src})")
    print("\nimages per split / label status:")
    print(pd.crosstab(df["split"], df["label_status"], margins=True).to_string())

    print("\nexample images with their label files:")
    for r in pd.concat([df[df.label_path != ""].head(5), df[df.label_path == ""].head(2)]).itertuples():
        print(f"  {Path(r.image_path).name:35s} -> {Path(r.label_path).name if r.label_path else '(no label file)'}")

    print("\n" + "=" * 70)
    print("STEP 2 - annotation inspection")
    print("=" * 70)
    labeled = df[df.label_status == "labeled"]
    zero = int((labeled.n_objects == 0).sum())
    cls_cols = ["n_D00", "n_D10", "n_D20", "n_D40", "n_other"]
    obj_counts = labeled[cls_cols].sum().rename(lambda c: c.replace("n_", ""))
    img_counts = (labeled[cls_cols] > 0).sum().rename(lambda c: c.replace("n_", ""))
    tab = pd.DataFrame({"objects": obj_counts, "images_containing": img_counts})
    tab.index.name = "damage_class"
    print("\ncounts per damage class (D40 = pothole; D00/D10/D20 = cracks):")
    print(tab.to_string())
    print(f"\nlabel files parsed                       : {len(labeled)}")
    print(f"label files with ZERO objects (empty)    : {zero}")
    print(f"images with no label file, assumed empty : {(df.label_status == 'empty_assumed').sum()}")
    print(f"images with no label file, UNLABELED     : {(df.label_status == 'unlabeled').sum()}  (excluded)")

    print("\nimage-level label preview (guide Step 3 rule):")
    prev = df["image_label"].value_counts()
    print(prev.to_string())
    n_norm = int(prev.get("normal", 0))
    print(f"\n-> images with no annotations ('normal'): {n_norm}")
    if n_norm < C.MIN_NORMAL_IMAGES:
        print(f"   fewer than {C.MIN_NORMAL_IMAGES}: Step 3 will switch to PATCH mode (guide rule).")
    else:
        print("   enough: Step 3 will use IMAGE-LEVEL labels (guide preference).")

    # save outputs ----------------------------------------------------------
    df.to_csv(C.PROC / "rdd_index.csv", index=False)
    tab.to_csv(C.RESULTS / "rdd_annotation_counts.csv")
    summary = pd.DataFrame({
        "metric": ["images_for_country", "labeled_files", "zero_object_label_files",
                   "no_label_file_assumed_empty", "no_label_file_unlabeled",
                   "images_pothole", "images_crack", "images_normal", "images_other_only"],
        "value": [len(df), len(labeled), zero,
                  int((df.label_status == "empty_assumed").sum()), int((df.label_status == "unlabeled").sum()),
                  int(prev.get("pothole", 0)), int(prev.get("crack", 0)), n_norm, int(prev.get("other_only", 0))],
    })
    summary.to_csv(C.RESULTS / "rdd_inspection_summary.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    tab["objects"].plot.bar(ax=ax[0], color="#4C72B0")
    ax[0].set_title("Boxes per damage class")
    ax[0].set_ylabel("boxes")
    prev.drop(labels=["unlabeled"], errors="ignore").plot.bar(ax=ax[1], color="#55A868")
    ax[1].set_title("Images per image-level label")
    ax[1].set_ylabel("images")
    for x in ax:
        x.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.savefig(C.RESULTS / "rdd_annotation_counts.png", dpi=150)
    plt.close()
    print(f"\nsaved: {C.PROC / 'rdd_index.csv'}")
    print(f"saved: {C.RESULTS / 'rdd_annotation_counts.csv'}, rdd_inspection_summary.csv, rdd_annotation_counts.png")
    print("\nDONE (Step 1 + 2). Check the class-id mapping above is right before continuing.")


if __name__ == "__main__":
    main()
