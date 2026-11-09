"""Shared helpers for the image pipeline (parsing, splitting, embedding)."""
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ----------------------------------------------------------------------------
# file discovery
# ----------------------------------------------------------------------------
def list_images(root):
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)


def detect_split(path, root):
    """train / val / test from folder names between root and the file."""
    parts = [p.lower() for p in Path(path).relative_to(root).parts[:-1]]
    for p in parts:
        if p in ("train", "training") or p.endswith("train"):
            return "train"
        if p in ("val", "valid", "validation") or p.endswith(("val", "valid", "validation")):
            return "val"
        if p in ("test", "testing") or p.endswith("test"):
            return "test"
    return "unknown"

def is_country(path, root, country):
    """True if the file belongs to `country` (name prefix India_xxx or a folder called India)."""
    if not country or country.lower() == "all":
        return True
    c = country.lower()
    folders = [p.lower() for p in Path(path).relative_to(root).parts[:-1]]
    return Path(path).stem.lower().startswith(c + "_") or c in folders


# ----------------------------------------------------------------------------
# annotations (YOLO txt or Pascal VOC xml - auto detected)
# ----------------------------------------------------------------------------
def read_yaml_names(root):
    """Try to read class names from a data.yaml shipped with the dataset."""
    for y in list(Path(root).rglob("*.yaml")) + list(Path(root).rglob("*.yml")):
        try:
            txt = y.read_text(errors="ignore")
        except Exception:
            continue
        m = re.search(r"names\s*:\s*\[(.*?)\]", txt, re.S)
        if m:
            names = [n.strip().strip("'\"") for n in m.group(1).split(",") if n.strip()]
            return {i: n for i, n in enumerate(names)}, y
        m = re.search(r"names\s*:\s*\n((?:\s+\d+\s*:\s*\S+\s*\n?)+)", txt)
        if m:
            d = {}
            for line in m.group(1).splitlines():
                mm = re.match(r"\s*(\d+)\s*:\s*(\S+)", line)
                if mm:
                    d[int(mm.group(1))] = mm.group(2).strip("'\"")
            return d, y
    return None, None


def canon(name):
    """Map any raw class name to D00/D10/D20/D40 or 'other'."""
    n = str(name).strip().upper()
    return n if n in {"D00", "D10", "D20", "D40"} else "other"


def parse_yolo(path, class_names):
    """-> list of (class, (cx,cy,w,h) normalised). Polygon lines are reduced to a bbox."""
    objs = []
    for line in Path(path).read_text(errors="ignore").splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        try:
            cid = int(float(p[0]))
            nums = list(map(float, p[1:]))
        except ValueError:
            continue
        if len(nums) == 4:
            cx, cy, w, h = nums
        else:  # segmentation polygon x1 y1 x2 y2 ...
            xs, ys = nums[0::2], nums[1::2]
            cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            w, h = max(xs) - min(xs), max(ys) - min(ys)
        objs.append((canon(class_names.get(cid, f"cls{cid}")), (cx, cy, w, h)))
    return objs


def parse_voc(path):
    """Pascal VOC via ElementTree -> list of (class, (xmin,ymin,xmax,ymax) pixels)."""
    root = ET.parse(path).getroot()
    objs = []
    for o in root.findall("object"):
        name = o.findtext("name", default="")
        bb = o.find("bndbox")
        if bb is None:
            continue
        box = tuple(float(bb.findtext(k, "0")) for k in ("xmin", "ymin", "xmax", "ymax"))
        objs.append((canon(name), box))
    return objs


def parse_label_file(path, kind, class_names):
    return parse_voc(path) if kind == "voc" else parse_yolo(path, class_names)


def boxes_in_pixels(objs, kind, width, height):
    """Return [(class, (x1,y1,x2,y2))] in pixels, clipped to the image."""
    out = []
    for cls, b in objs:
        if kind == "yolo":
            cx, cy, w, h = b
            x1, y1, x2, y2 = (cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height
        else:
            x1, y1, x2, y2 = b
        x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
        y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
        if x2 > x1 and y2 > y1:
            out.append((cls, (x1, y1, x2, y2)))
    return out


def image_label_from_classes(classes):
    """Guide Step 3 image-level rule. classes = iterable of canonical names in the image."""
    s = set(classes)
    if C.POTHOLE_CLASS in s:
        return "pothole"
    if s & C.CRACK_CLASSES:
        return "crack"
    if not s:
        return "normal"
    return "other_only"   # only classes outside D00/D10/D20/D40 -> excluded


# ----------------------------------------------------------------------------
# RDD2022 index (one row per image)
# ----------------------------------------------------------------------------
def build_rdd_index(rdd_root, country=C.COUNTRY, yolo_classes=None, missing_policy="auto"):
    """
    Scan rdd_root, attach label files (txt or xml, matched by file stem),
    parse them, and decide each image's label status.

    label_status:
        labeled        a label file exists
        empty_assumed  no label file, but most images of that split have one
                       (background images stored without a txt)   [policy auto/empty]
        unlabeled      no label file and the split has no labels (e.g. RDD2022 test)
    """
    rdd_root = Path(rdd_root)
    if not rdd_root.exists():
        raise SystemExit(f"[error] RDD2022 folder not found: {rdd_root}\n"
                         f"        Put the Kaggle download there or pass --rdd-root")

    names, yaml_path = read_yaml_names(rdd_root)
    class_names = dict(yolo_classes or names or C.YOLO_CLASS_NAMES)
    src = "CLI" if yolo_classes else (f"yaml:{yaml_path}" if names else "default D00,D10,D20,D40")

    all_imgs = list_images(rdd_root)
    lab_files = {}
    for p in rdd_root.rglob("*"):
        if p.suffix.lower() in (".txt", ".xml"):
            lab_files.setdefault(p.stem, []).append(p)

    rows = []
    for ip in all_imgs:
        if not is_country(ip, rdd_root, country):
            continue
        split = detect_split(ip, rdd_root)
        cands = lab_files.get(ip.stem, [])
        lp = None
        if cands:
            cands = sorted(cands, key=lambda q: (q.suffix.lower() != ".xml",
                                                 not any(s in str(q).lower() for s in ("label", "annot"))))
            lp = cands[0]
        rows.append(dict(image_id=ip.stem, image_path=str(ip), split=split,
                         label_path=str(lp) if lp else "",
                         kind=("voc" if lp and lp.suffix.lower() == ".xml" else "yolo" if lp else "none")))
    df = pd.DataFrame(rows)
    if df.empty:
        return df, class_names, src

    split_frac = df.groupby("split")["label_path"].apply(lambda s: (s != "").mean()).to_dict()
    status = []
    for r in df.itertuples():
        if r.label_path:
            status.append("labeled")
        elif missing_policy == "empty":
            status.append("empty_assumed")
        elif missing_policy == "auto" and split_frac.get(r.split, 0) >= 0.5:
            status.append("empty_assumed")
        else:
            status.append("unlabeled")
    df["label_status"] = status

    names_per_img = []
    for r in df.itertuples():
        objs = []
        if r.label_status == "labeled":
            try:
                objs = parse_label_file(r.label_path, r.kind, class_names)
            except Exception as e:
                print(f"[warn] could not parse {r.label_path}: {e}")
        names_per_img.append([o[0] for o in objs])
    for c in ["D00", "D10", "D20", "D40", "other"]:
        df[f"n_{c}"] = [lst.count(c) for lst in names_per_img]
    df["n_objects"] = [len(l) for l in names_per_img]
    df["image_label"] = ["unlabeled" if st == "unlabeled" else image_label_from_classes(l)
                         for st, l in zip(df["label_status"], names_per_img)]
    return df, class_names, src


# ----------------------------------------------------------------------------
# splitting
# ----------------------------------------------------------------------------
def best_group_split(df, label_col="label", group_col="group", sizes=C.SPLIT, seed=C.SEED, tries=40):
    """
    70/15/15 split by GROUP (image id) with GroupShuffleSplit, so crops of one
    photo never straddle splits. GroupShuffleSplit is not stratified, so several
    seeds are tried and the split whose class proportions are closest to the
    overall ones is kept. Returns an array of 'train'/'val'/'test'.
    """
    from sklearn.model_selection import GroupShuffleSplit
    tr, va, te = sizes
    overall = df[label_col].value_counts(normalize=True)
    best, best_score = None, 1e9
    for k in range(tries):
        gss1 = GroupShuffleSplit(n_splits=1, test_size=te, random_state=seed + k)
        idx_trval, idx_te = next(gss1.split(df, groups=df[group_col]))
        sub = df.iloc[idx_trval]
        gss2 = GroupShuffleSplit(n_splits=1, test_size=va / (tr + va), random_state=seed + 100 + k)
        i_tr, i_va = next(gss2.split(sub, groups=sub[group_col]))
        lab = np.array([""] * len(df), dtype=object)
        lab[idx_trval[i_tr]] = "train"
        lab[idx_trval[i_va]] = "val"
        lab[idx_te] = "test"
        score, ok = 0.0, True
        for s in ("train", "val", "test"):
            part = df[lab == s][label_col].value_counts(normalize=True).reindex(overall.index).fillna(0)
            if (part == 0).any():
                ok = False
            score += (part - overall).abs().max()
        if not ok:
            score += 10
        if score < best_score:
            best, best_score = lab, score
    return best


# ----------------------------------------------------------------------------
# embeddings (MobileNetV2, 1280-d)
# ----------------------------------------------------------------------------
def build_backbone(pretrained=True, state_dict=None):
    import torch
    from torchvision import models
    m = models.mobilenet_v2(weights="DEFAULT" if (pretrained and state_dict is None) else None)
    m.classifier = torch.nn.Identity()          # drop classifier, keep pooled 1,280-d output
    if state_dict is not None:
        m.load_state_dict(state_dict)
    m.eval()
    return m


def make_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((C.IMG_SIZE, C.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD),
    ])


def embed_paths(paths, backbone, batch_size=C.BATCH_SIZE, pil_transform=None, log_every=10):
    """
    Embed image files on CPU in batches.
    Returns (X float32 [n_ok,1280], ok_index = positions that were readable).
    pil_transform: optional PIL->PIL function applied first (augmentation experiments).
    """
    import torch
    from PIL import Image
    tfm = make_transform()
    feats, ok, batch, batch_idx = [], [], [], []
    state = {"nb": 0}

    def flush():
        if not batch:
            return
        with torch.no_grad():
            out = backbone(torch.stack(batch)).numpy().astype(np.float32)
        feats.append(out)
        ok.extend(batch_idx)
        batch.clear()
        batch_idx.clear()
        state["nb"] += 1
        if state["nb"] % log_every == 0:
            print(f"    embedded ~{len(ok)}/{len(paths)}")

    for i, p in enumerate(paths):
        try:
            img = Image.open(p).convert("RGB")
            if pil_transform is not None:
                img = pil_transform(img)
            batch.append(tfm(img))
            batch_idx.append(i)
        except Exception as e:
            print(f"[warn] skipped unreadable image {p}: {e}")
            continue
        if len(batch) == batch_size:
            flush()
    flush()
    X = np.concatenate(feats) if feats else np.zeros((0, 1280), np.float32)
    return X, ok


# ----------------------------------------------------------------------------
# misc
# ----------------------------------------------------------------------------
def bootstrap_ci(correct, n_boot=2000, seed=C.SEED, alpha=0.05):
    """Percentile bootstrap CI for the mean of a 0/1 vector (accuracy)."""
    correct = np.asarray(correct, dtype=float)
    n = len(correct)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    stats = np.array([correct[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return float(correct.mean()), float(np.percentile(stats, 100 * alpha / 2)), float(np.percentile(stats, 100 * (1 - alpha / 2)))


def ensure_dirs():
    for d in (C.PROC, C.RESULTS, C.MODELS_DIR, C.DEMO_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)
