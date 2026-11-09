"""
Run the whole Part 2 pipeline in guide order (Steps 1-8 + report tables).

    python src/image/run_all.py
    python src/image/run_all.py --rdd-root D:/RDD2022 --alley-root D:/AlleyFloodNet
    python src/image/run_all.py --skip-flood          # guide fallback: no flood class
    python src/image/run_all.py --with-augmentation   # adds optional Step 9

It stops at the first failing step so you can fix it and re-run (each step is also
runnable alone; they only read the previous step's files).
"""
import argparse
import subprocess
import sys
from pathlib import Path

import config as C

HERE = Path(__file__).resolve().parent


def run(script, *args):
    cmd = [sys.executable, str(HERE / script), *args]
    print("\n" + "#" * 78 + f"\n# {script} {' '.join(args)}\n" + "#" * 78, flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"\n[stopped] {script} failed (exit {r.returncode}). Fix it, then re-run.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rdd-root", default=str(C.RDD_ROOT))
    ap.add_argument("--alley-root", default=str(C.ALLEY_ROOT))
    ap.add_argument("--country", default=C.COUNTRY)
    ap.add_argument("--yolo-classes", default=None)
    ap.add_argument("--missing-labels", default="auto", choices=["auto", "empty", "skip"])
    ap.add_argument("--mode", default="auto", choices=["auto", "image", "patch"])
    ap.add_argument("--skip-flood", action="store_true")
    ap.add_argument("--flood2-root", default=None)
    ap.add_argument("--with-augmentation", action="store_true")
    ap.add_argument("--only-correct-demo", action="store_true")
    ap.add_argument("--test-mode", action="store_true", help="smoke test with random weights (never for real results)")
    a = ap.parse_args()

    common = ["--rdd-root", a.rdd_root, "--country", a.country, "--missing-labels", a.missing_labels]
    if a.yolo_classes:
        common += ["--yolo-classes", a.yolo_classes]

    run("s1_s2_inspect_rdd.py", *common)
    run("s3_build_rdd_dataset.py", *common, "--mode", a.mode)
    run("s4_prepare_flood.py", *(["--skip-flood"] if a.skip_flood else ["--alley-root", a.alley_root]))
    run("s5_extract_embeddings.py", *(["--no-pretrained"] if a.test_mode else []))
    run("s6_train_models.py", *(["--allow-random-weights"] if a.test_mode else []))
    if not a.skip_flood:
        run("s7_flood_crossset.py", *(["--flood2-root", a.flood2_root] if a.flood2_root else []))
    run("s8_package_demo.py", *(["--only-correct"] if a.only_correct_demo else []))
    if a.with_augmentation:
        run("s9_augmentation_experiment.py")
    run("s10_report_material.py")
    print("\nALL STEPS FINISHED. Review results/image/, then follow the git steps.")


if __name__ == "__main__":
    main()
