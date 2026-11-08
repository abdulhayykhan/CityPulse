"""
Guide Part 2, Step 4 - flood images (AlleyFloodNet from Kaggle)

AlleyFloodNet layout:  train/{flooded,non_flooded}  and  test/{flooded,non_flooded}

What this does
  1. Puts every AlleyFloodNet image into  data/processed/image/flood_sources/alleyfloodnet/{flood,non_flood}/
     (the 'flood/ and non_flood/ folders' the guide asks for; original split kept in the file name).
  2. Adds them to the classifier dataset:
        flooded      -> class 'flood'
        non_flooded  -> class 'normal'   (guide: "non-flood images can be added to the normal class")
     Original AlleyFloodNet TEST  -> our test split.
     Original AlleyFloodNet TRAIN -> our train (85%) / val (15%), stratified.
  3. Writes manifest.csv (RDD + flood) and the final class-count table + figure.

Fallback (guide): if you cannot use the flood data, run with --skip-flood. The classifier
then has pothole / crack / normal only; tell the team.

LICENSE: AlleyFloodNet is a research dataset. Check the licence on its Kaggle page /
paper before sharing images. Raw images are never committed to git anyway; only the
10 demo photos are.
"""
import argparse
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

import common as cm
import config as C


def flood_class_of(path):
    """flood / non_flood from the parent folder name."""
    name = path.parent.name.lower().replace("-", "_").replace(" ", "_")
    if "non" in name or "no_flood" in name or name in ("dry", "normal", "not_flooded"):
        return "non_flood"
    if "flood" in name:
        return "flood"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alley-root", default=str(C.ALLEY_ROOT))
    ap.add_argument("--skip-flood", action="store_true", help="drop the flood class (guide fallback)")
    ap.add_argument("--seed", type=int, default=C.SEED)
    a = ap.parse_args()

    cm.ensure_dirs()
    rdd_manifest = C.PROC / "manifest_rdd.csv"
    if not rdd_manifest.exists():
        raise SystemExit("[error] run s3_build_rdd_dataset.py first")
    rdd = pd.read_csv(rdd_manifest)

    if a.skip_flood:
        print("--skip-flood: flood class dropped, classes = pothole / crack / normal")
        man = rdd.copy()
    else:
        root = Path(a.alley_root)
        if not root.exists():
            raise SystemExit(f"[error] AlleyFloodNet folder not found: {root}\n"
                             "        download it or pass --alley-root, or use --skip-flood")
        imgs = cm.list_images(root)
        rows = []
        for p in imgs:
            c = flood_class_of(p)
            if c is None:
                continue
            rows.append(dict(path=p, cls=c, orig_split=cm.detect_split(p, root)))
        al = pd.DataFrame(rows)
        if al.empty:
            raise SystemExit("[error] found no images under flooded/ or non_flooded/ folders in " + str(root))
        print("AlleyFloodNet images by original split and class:")
        print(pd.crosstab(al.orig_split, al.cls, margins=True).to_string())

        # 1) flood/ and non_flood/ source folders ---------------------------------
        src_root = C.FLOOD_SRC_DIR / "alleyfloodnet"
        if src_root.exists():
            shutil.rmtree(src_root)
        for i, r in enumerate(al.itertuples()):
            dst = src_root / r.cls / f"{r.orig_split}_{i:05d}{r.path.suffix.lower()}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(r.path, dst)

        # 2) assign to our splits ----------------------------------------------
        al = al.reset_index(drop=True)
        al["split"] = ""
        is_test = al.orig_split == "test"
        al.loc[is_test, "split"] = "test"
        rest = al[~is_test]
        if len(rest):
            tr_idx, va_idx = train_test_split(rest.index, test_size=0.15, stratify=rest.cls, random_state=a.seed)
            al.loc[tr_idx, "split"] = "train"
            al.loc[va_idx, "split"] = "val"
        if (al.split == "").any():
            raise SystemExit("[error] could not assign splits for some AlleyFloodNet images")

        # remove a previous run of this step, then copy
        for old in C.DATASET_DIR.rglob("alley_*"):
            old.unlink()
        recs = []
        for i, r in enumerate(al.itertuples()):
            label = "flood" if r.cls == "flood" else "normal"
            name = f"alley_{r.orig_split}_{i:05d}{r.path.suffix.lower()}"
            rel = f"{r.split}/{label}/{name}"
            dst = C.DATASET_DIR / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(r.path, dst)
            recs.append(dict(rel_path=rel, split=r.split, label=label, source="alleyfloodnet",
                             group=f"alley_{i:05d}", origin_path=str(r.path), orig_split=r.orig_split,
                             box="", mode="image"))
        man = pd.concat([rdd, pd.DataFrame(recs)], ignore_index=True)

    man.to_csv(C.PROC / "manifest.csv", index=False)

    # counts -------------------------------------------------------------------
    counts = pd.crosstab(man["label"], man["split"]).reindex(columns=["train", "val", "test"]).fillna(0).astype(int)
    counts["total"] = counts.sum(axis=1)
    counts.loc["TOTAL"] = counts.sum()
    counts.to_csv(C.RESULTS / "dataset_counts.csv")
    by_src = pd.crosstab([man["source"], man["label"]], man["split"]).reindex(columns=["train", "val", "test"]).fillna(0).astype(int)
    by_src.to_csv(C.RESULTS / "dataset_counts_by_source.csv")

    print("\nFINAL class counts per split:")
    print(counts.to_string())
    print("\nby source:")
    print(by_src.to_string())

    plot = counts.drop(index="TOTAL")[["train", "val", "test"]]
    ax = plot.plot.bar(stacked=True, figsize=(7, 4), color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_ylabel("samples")
    ax.set_title("Image classifier dataset: samples per class and split")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.savefig(C.RESULTS / "dataset_counts.png", dpi=150)
    plt.close()
    print(f"\nsaved manifest: {C.PROC / 'manifest.csv'}")
    print("DONE (Step 4). Next: python src/image/s5_extract_embeddings.py")


if __name__ == "__main__":
    main()
