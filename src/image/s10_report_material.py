"""
Guide Part 2 - "Report material you hand over"
Collects the result CSVs into two ready-to-paste Markdown files:

  results/image/metrics_for_group_chat.md   short table to post in the group after the model
  results/image/report_material.md          dataset counts, SVM-vs-RF table, per-class F1,
                                            confusion matrix, flood cross-set result with CI,
                                            and a limitations paragraph filled with your real numbers

Run after s6 and s7.
"""
import json

import pandas as pd

import config as C


def md(df, floatfmt=3):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for v in r:
            cells.append(f"{v:.{floatfmt}f}" if isinstance(v, float) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def read(name, **kw):
    p = C.RESULTS / name
    return pd.read_csv(p, **kw) if p.exists() else None


def main():
    summary, per = read("metrics_summary.csv"), read("per_class_metrics.csv")
    counts, by_src = read("dataset_counts.csv", index_col=0), read("dataset_counts_by_source.csv")
    flood, src_test = read("flood_crossset_results.csv"), read("test_by_source.csv")
    mode = json.loads((C.PROC / "dataset_mode.json").read_text()) if (C.PROC / "dataset_mode.json").exists() else {}
    prm = json.loads((C.RESULTS / "best_params.json").read_text()) if (C.RESULTS / "best_params.json").exists() else {}
    if summary is None:
        raise SystemExit("[error] run s6_train_models.py first")

    test = summary[summary.split == "test"][["model", "n", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]]
    val = summary[summary.split == "val"][["model", "n", "accuracy", "macro_f1"]]
    best_name = summary[summary.split == "val"].sort_values("macro_f1", ascending=False).iloc[0].model

    # ------------------------------------------------ group chat
    chat = ["**CityPulse - Image model metrics (Bushra)**",
            f"Features: MobileNetV2 embeddings (1280-d). Split by image id 70/15/15. Dataset mode: {mode.get('mode', '?')}.",
            "", "Test set:", md(test), "", f"Packaged model (best validation macro-F1): **{best_name}** -> models/image_model.joblib",
            "Interface: `predict_image(path) -> {type, confidence}`"]
    if flood is not None:
        a = flood[flood.experiment.str.startswith("A")].iloc[0]
        chat += ["", f"Flood (AlleyFloodNet in-dataset): accuracy {a.accuracy:.3f}, 95% CI [{a.ci95_low:.3f}, {a.ci95_high:.3f}], n={a.n_test}"]
    (C.RESULTS / "metrics_for_group_chat.md").write_text("\n".join(chat) + "\n", encoding="utf-8")

    # ------------------------------------------------ report
    R = ["# Image component - results (Bushra Zafar)", ""]
    R += ["## 1. Dataset counts", "",
          f"Mode: **{mode.get('mode', '?')}**-level samples; cap {mode.get('cap', '?')} per class; split 70/15/15 by image id.", ""]
    if counts is not None:
        R += [md(counts.reset_index().rename(columns={"label": "class"}), 0), ""]
    if by_src is not None:
        R += ["By source:", "", md(by_src, 0), ""]
    R += ["## 2. SVM vs Random Forest", "", "Test:", "", md(test), "", "Validation:", "", md(val), ""]
    if prm:
        R += [f"Hyper-parameters (grouped {prm.get('cv', 'CV')}, macro-F1): SVM {prm.get('svm')}, RF {prm.get('rf')}", ""]
    t = per[per.split == "test"].pivot(index="class", columns="model", values="f1").reset_index()
    R += ["Per-class F1 (test):", "", md(t), "", "Per-class precision / recall / F1 (test):", "",
          md(per[per.split == "test"][["model", "class", "precision", "recall", "f1", "support"]]), ""]
    if src_test is not None:
        R += ["Test accuracy by data source:", "", md(src_test.drop(columns=["classes_present"], errors="ignore")), ""]
    R += ["## 3. Confusion matrix", "",
          "See `confusion_matrix_svm_test.png` and `confusion_matrix_rf_test.png` (counts and row-normalised).", ""]
    R += ["## 4. Flood cross-set result", ""]
    if flood is not None:
        R += [md(flood[["experiment", "train_set", "test_set", "n_test", "accuracy", "ci95_low", "ci95_high", "flood_recall", "false_alarm_rate"]]), "",
              "95% percentile bootstrap CI over test images (2,000 resamples).", ""]
        if not flood.experiment.str.startswith("C").any():
            R += ["**Only one flood dataset was available**, so the two-way cross-set test of the guide (train on one dataset, "
                  "test on the other) was not possible. Experiment A is an in-dataset result and Experiment B is an external "
                  "false-alarm test on flood-free RDD2022 road photos; neither measures recall on a second flood source.", ""]
    else:
        R += ["_s7_flood_crossset.py not run yet._", ""]

    n_flood = int(counts.loc["flood", "total"]) if counts is not None and "flood" in counts.index else 0
    R += ["## 5. Limitations (draft paragraph)", "",
          "The image classifier uses frozen ImageNet MobileNetV2 embeddings with an SVM or Random Forest; it was not fine-tuned. "
          "Road-damage labels come from RDD2022 (India subset, collected from smartphone-mounted vehicles in India), so results do not "
          "describe Karachi roads. "
          + ("Image-level labels were derived from bounding-box annotations (pothole if any D40 box, crack if only D00/D10/D20 boxes, "
             "normal if no annotations); this is a proxy label, and an image marked 'normal' may still contain damage nobody annotated. "
             if mode.get("mode") == "image" else
             "Because too few un-annotated photos existed, the classifier is trained on cropped patches (pothole and crack boxes of at "
             "least 32 px plus random patches that overlap no box); 'normal' patches are a proxy label and the app, which sends whole "
             "photos, sees a different input distribution than the patches used in training. ")
          + f"The flood class comes from a single, small dataset (AlleyFloodNet, {n_flood} flood images in this experiment) and is "
          "collected in a different setting from the road-damage photos, so the classifier could partly learn dataset identity rather than "
          "flooding; the non-flooded images were added to the 'normal' class to reduce this. Flood accuracy has a wide bootstrap confidence "
          "interval and a true two-dataset cross-set test was limited by data availability. Predicted confidences are relative scores, "
          "not calibrated probabilities. Reported performance is on held-out public images, not on photos taken by citizens with their own phones.", ""]
    (C.RESULTS / "report_material.md").write_text("\n".join(R), encoding="utf-8")
    print("saved:", C.RESULTS / "metrics_for_group_chat.md")
    print("saved:", C.RESULTS / "report_material.md")
    print("\n" + "\n".join(chat))


if __name__ == "__main__":
    main()
