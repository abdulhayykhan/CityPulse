# CityPulse - Part 2: Image classifier (Bushra Zafar)

Output required by the guide: `predict_image(path)`, a saved model, a metrics table, and a flood cross-set result.

| Guide step | Script | Main outputs |
|---|---|---|
| 1 + 2 List India images, inspect annotations | `s1_s2_inspect_rdd.py` | `results/image/rdd_annotation_counts.{csv,png}`, `rdd_inspection_summary.csv` |
| 3 Build classification dataset | `s3_build_rdd_dataset.py` | `data/processed/image/dataset/{train,val,test}/<class>/`, `dataset_counts_rdd.csv` |
| 4 Flood images | `s4_prepare_flood.py` | `flood_sources/alleyfloodnet/{flood,non_flood}/`, `dataset_counts.{csv,png}` |
| 5 Embeddings (MobileNetV2, 1280-d) | `s5_extract_embeddings.py` | `X_{train,val,test}.npy` + labels + meta |
| 6 SVM vs Random Forest | `s6_train_models.py` | `metrics_summary.csv`, `per_class_metrics.csv`, confusion matrices, `models/image_model.joblib` |
| 7 Flood cross-set + bootstrap CI | `s7_flood_crossset.py` | `flood_crossset_results.csv`, `flood_crossset_ci.png` |
| 8 Package for Zainab | `s8_package_demo.py` + `src/models/image.py` | `demo_samples/images/` (10 photos) + `results/image/demo_manifest.csv` |
| 9 Optional augmentation | `s9_augmentation_experiment.py` | `augmentation_experiment.csv` |
| Report material | `s10_report_material.py` | `report_material.md`, `metrics_for_group_chat.md` |

## Where to put the data (never committed)

```
data/raw/rdd2022/          # Kaggle RDD2022 (train/ val/ test/ with images + labels)
data/raw/alleyfloodnet/    # Kaggle AlleyFloodNet (train/ test/ with flooded/ non_flooded/)
```
Other locations work too: pass `--rdd-root` / `--alley-root`.

## Run

```bash
conda activate citypulse
python src/image/run_all.py                       # everything, in guide order
python src/image/run_all.py --skip-flood          # guide fallback: no flood class
python src/image/run_all.py --flood2-root PATH    # 2nd flood dataset (Mendeley images/labels or flood/ + non_flood/)
python src/image/run_all.py --with-augmentation   # adds optional Step 9
```
Or run the steps one by one (`python src/image/s1_s2_inspect_rdd.py`, ...). Each step only needs the previous step's files and prints its own "Done when" check.

Pre-flight check without real data: `python src/image/make_fake_data.py --out /tmp/fake` then
`python src/image/run_all.py --rdd-root /tmp/fake/rdd2022 --alley-root /tmp/fake/alleyfloodnet --test-mode`
(`--test-mode` uses random weights: numbers are meaningless, delete the outputs afterwards).

## How your Kaggle copy of RDD2022 is handled

* **No XML**: labels are read as YOLO `.txt` (`class cx cy w h`, normalised). Pascal VOC `.xml` is also auto-detected if it exists.
* **Class ids**: read from `data.yaml` if the dataset has one, else assumed `0=D00, 1=D10, 2=D20, 3=D40`. **Step 1 prints the mapping it used - check it.** Override with `--yolo-classes D00,D10,D20,D40`.
* **No country folders**: India = file names starting with `India_` (or a folder called India). `--country all` disables the filter (do not do this without telling the team).
* **Splits**: your train/val/test folders are pooled and re-split 70/15/15 by image id, as the guide requires. The original RDD2022 test set normally has no labels; those images are excluded (never treated as "normal").
* **Images without a label file**: if most images of a split have label files, a missing file means "background image" (-> normal). If a split has no label files at all, those images are "unlabeled" and excluded. Override with `--missing-labels empty|skip`.
* **Image mode vs patch mode**: image-level labels if >= 300 normal images exist, otherwise patches (guide rule). Force with `--mode image|patch`. The mode is printed by Step 3 and stored in `data/processed/image/dataset_mode.json`.

## Design decisions (so you can defend them in the viva)

* **Split by image id** (`GroupShuffleSplit`) and **grouped cross-validation** (`GroupKFold`) so crops/near-copies of one photo never sit in train and test.
* The model is fitted on **train only**; validation picks the winner (SVM vs RF); test is reported once. The saved model is exactly the one the table describes.
* **Predictions = argmax of `predict_proba`** (what `predict_image` returns), so the tables match deployed behaviour.
* **flood class** = AlleyFloodNet `flooded`; AlleyFloodNet `non_flooded` joins **normal** (guide). Because flood images come from one source, the script reports accuracy **per source** (`test_by_source.csv`) so you can show the classifier is not just recognising the dataset.
* **Flood cross-set**: the guide's two-way test needs two flood datasets. With only AlleyFloodNet the script runs (A) in-dataset official train->test and (B) an external false-alarm test on flood-free RDD2022 road photos. Add `--flood2-root` when you get the second dataset to run the real two-way test (C). The report text says which ones were run.
* **Confidence** is the top class probability (Platt-scaled for the SVM); it is a relative score, not a calibrated guarantee.
* `models/image_model.joblib` is self-contained: it embeds the MobileNetV2 weights, so Zainab's machine needs no internet and no torch cache.

## Interface (fixed contract)

```python
from src.models.image import predict_image
predict_image("demo_samples/images/pothole_01.jpg")   # {'type': 'pothole', 'confidence': 0.93}
```
`type` is one of `pothole`, `crack`, `normal`, `flood` (no `flood` if you used `--skip-flood`). Tell Zainab the exact strings. Run from the repo root.

## Things to report honestly
* Patch mode (if it was used) trains on crops but the app sends whole photos: `results/image/whole_photo_check_patch_mode.csv` measures this.
* Flood data is small and from one source; the bootstrap CI shows how uncertain the numbers are.
* Image-level labels are a proxy built from box annotations (an un-annotated image may still contain damage).
* Public India data, not Karachi roads and not citizen phone photos.
