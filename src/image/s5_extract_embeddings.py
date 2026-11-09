"""
Guide Part 2, Step 5 - embeddings with a pretrained MobileNetV2 (fixed feature extractor)

torchvision.models.mobilenet_v2(weights='DEFAULT'), classifier removed, pooled
1,280-number output, 224x224 resize, ImageNet mean/std, CPU, batches.

Saves (in data/processed/image/):
    X_train.npy X_val.npy X_test.npy      float32 [n, 1280]
    y_train.npy y_val.npy y_test.npy      class names
    meta_train.csv meta_val.csv meta_test.csv   rel_path, label, source, group, orig_split (row-aligned)
    backbone_mobilenet_v2.pt              the exact weights used (packaged inside the joblib later)
    embedding_info.json

Done when: arrays load and their row counts match the labels  (checked at the end).

The first run downloads the ImageNet weights (~14 MB) to the torch cache - needs internet once.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

import common as cm
import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="TEST ONLY: random weights. Never use for real results.")
    a = ap.parse_args()

    if a.threads:
        torch.set_num_threads(a.threads)
    man_path = C.PROC / "manifest.csv"
    if not man_path.exists():
        raise SystemExit("[error] run s3 and s4 first (manifest.csv missing)")
    man = pd.read_csv(man_path)

    torch.manual_seed(C.SEED)
    backbone = cm.build_backbone(pretrained=not a.no_pretrained)
    torch.save(backbone.state_dict(), C.PROC / "backbone_mobilenet_v2.pt")
    print("backbone: MobileNetV2 -> output dim 1280 | pretrained =", not a.no_pretrained)

    t0 = time.time()
    for split in ("train", "val", "test"):
        part = man[man.split == split].reset_index(drop=True)
        paths = [str(C.DATASET_DIR / p) for p in part.rel_path]
        print(f"\n[{split}] {len(paths)} images")
        X, ok = cm.embed_paths(paths, backbone, batch_size=a.batch_size)
        part = part.iloc[ok].reset_index(drop=True)
        y = part["label"].to_numpy().astype(str)
        assert len(X) == len(y) == len(part), "row count mismatch"
        assert not np.isnan(X).any(), "NaN in embeddings"
        np.save(C.PROC / f"X_{split}.npy", X)
        np.save(C.PROC / f"y_{split}.npy", y)
        part.to_csv(C.PROC / f"meta_{split}.csv", index=False)
        print(f"[{split}] X {X.shape}  y {y.shape}  classes {pd.Series(y).value_counts().to_dict()}")

    info = dict(backbone="mobilenet_v2", pretrained=not a.no_pretrained, dim=1280, img_size=C.IMG_SIZE,
                mean=C.IMAGENET_MEAN, std=C.IMAGENET_STD, seconds=round(time.time() - t0, 1))
    (C.PROC / "embedding_info.json").write_text(json.dumps(info, indent=2))

    print("\nVERIFY (Done-when check):")
    for s in ("train", "val", "test"):
        X, y, m = np.load(C.PROC / f"X_{s}.npy"), np.load(C.PROC / f"y_{s}.npy"), pd.read_csv(C.PROC / f"meta_{s}.csv")
        print(f"  {s:5s}: X rows {len(X):5d} | labels {len(y):5d} | meta {len(m):5d} | match = {len(X) == len(y) == len(m)}")
    print(f"\nDONE (Step 5) in {info['seconds']} s. Next: python src/image/s6_train_models.py")


if __name__ == "__main__":
    main()
