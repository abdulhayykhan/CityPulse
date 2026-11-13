"""
SMOKE-TEST HELPER (not part of the real pipeline).
Creates tiny fake datasets in the SAME layout as your Kaggle downloads so you (or CI)
can check that every script runs before you spend time on the real data:

    python src/image/make_fake_data.py --out /tmp/fake
    python src/image/run_all.py --rdd-root /tmp/fake/rdd2022 --alley-root /tmp/fake/alleyfloodnet --test-mode

--test-mode uses random network weights, so the numbers mean nothing.
"""
import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def road(w=320, h=320, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(70, 110)
    arr = np.clip(rng.normal(base, 12, (h, w, 3)), 0, 255).astype("uint8")
    return Image.fromarray(arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fake_data")
    ap.add_argument("--n", type=int, default=240, help="India images")
    ap.add_argument("--empty-frac", type=float, default=0.30)
    a = ap.parse_args()
    rnd = random.Random(0)
    out = Path(a.out)

    # ---- RDD2022-like: train/val/test, images/ + labels/ (YOLO txt), country prefix
    k = 0
    for split, n in (("train", int(a.n * 0.7)), ("val", int(a.n * 0.2)), ("test", int(a.n * 0.1))):
        for sub in ("images", "labels"):
            (out / "rdd2022" / split / sub).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            for country in ("India", "Japan"):
                k += 1
                name = f"{country}_{k:06d}"
                im = road(seed=k)
                d = ImageDraw.Draw(im)
                lines = []
                kind = rnd.random()
                if kind < a.empty_frac:
                    pass                                           # background image
                elif kind < 0.65:                                    # pothole (class 3)
                    x, y, w, h = rnd.randint(30, 200), rnd.randint(30, 200), rnd.randint(50, 90), rnd.randint(50, 90)
                    d.ellipse([x, y, x + w, y + h], fill=(20, 20, 20))
                    lines.append(f"3 {(x + w / 2) / 320:.4f} {(y + h / 2) / 320:.4f} {w / 320:.4f} {h / 320:.4f}")
                else:                                                # crack (class 0/1/2)
                    x, y = rnd.randint(20, 150), rnd.randint(20, 250)
                    w, h = rnd.randint(100, 150), rnd.randint(40, 60)
                    d.line([x, y, x + w, y + h], fill=(15, 15, 15), width=3)
                    lines.append(f"{rnd.choice([0, 1, 2])} {(x + w / 2) / 320:.4f} {(y + h / 2) / 320:.4f} {w / 320:.4f} {h / 320:.4f}")
                im.save(out / "rdd2022" / split / "images" / f"{name}.jpg")
                if split != "test":                                  # original RDD2022 test set has no labels
                    (out / "rdd2022" / split / "labels" / f"{name}.txt").write_text("\n".join(lines))

    # ---- AlleyFloodNet-like: train/test x flooded/non_flooded
    for split, n in (("train", 60), ("test", 20)):
        for cls in ("flooded", "non_flooded"):
            p = out / "alleyfloodnet" / split / cls
            p.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                im = road(seed=1000 + i + (500 if cls == "flooded" else 0))
                if cls == "flooded":
                    d = ImageDraw.Draw(im)
                    d.rectangle([0, 160, 320, 320], fill=(60, 90, 140))
                im.save(p / f"{cls}_{split}_{i:03d}.jpg")
    print("fake data written to", out)


if __name__ == "__main__":
    main()
