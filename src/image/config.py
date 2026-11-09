"""
CityPulse - Part 2 (Image lead: Bushra Zafar)
Central configuration. Change paths here (or pass CLI flags) - nothing else
in the pipeline hard-codes a path.

Repo layout assumed (adjust if the proposal Section 15 differs):
    data/raw/rdd2022/          <- Kaggle RDD2022 (NOT committed, ignored by git)
    data/raw/alleyfloodnet/    <- Kaggle AlleyFloodNet (NOT committed)
    data/processed/image/      <- generated dataset + embeddings (NOT committed)
    results/image/             <- metric CSVs + PNG figures (committed)
    models/image_model.joblib  <- packaged model (committed)
    src/models/image.py        <- predict_image(path) (committed)
    demo_samples/images/       <- 10 demo photos (committed)
"""
from pathlib import Path

# src/image/config.py -> parents[0]=image, [1]=src, [2]=repo root
ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw"
RDD_ROOT = RAW_DIR / "rdd2022"
ALLEY_ROOT = RAW_DIR / "alleyfloodnet"

PROC = ROOT / "data" / "processed" / "image"
DATASET_DIR = PROC / "dataset"          # train/ val/ test/ <class>/ images
FLOOD_SRC_DIR = PROC / "flood_sources"  # flood/ and non_flood/ (guide Step 4)
RESULTS = ROOT / "results" / "image"
MODELS_DIR = ROOT / "models"
DEMO_DIR = ROOT / "demo_samples" / "images"

# ---- RDD2022 ---------------------------------------------------------------
COUNTRY = "India"                       # guide: India subset only
YOLO_CLASS_NAMES = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}   # override via data.yaml / CLI
CRACK_CLASSES = {"D00", "D10", "D20"}
POTHOLE_CLASS = "D40"

# ---- Dataset building --------------------------------------------------------
MIN_NORMAL_IMAGES = 300     # guide: "fewer than a few hundred" -> switch to patches
CAP_PER_CLASS = 1500        # guide: cap each class at about 1,500 samples
MIN_BOX_PX = 32             # guide: skip boxes smaller than 32 px
SPLIT = (0.70, 0.15, 0.15)  # train / val / test, split by image id
SEED = 42

# ---- Embeddings (guide Step 5) ---------------------------------------------
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
BATCH_SIZE = 32

CLASS_ORDER = ["pothole", "crack", "normal", "flood"]   # display order only
