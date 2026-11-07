# Notebook Rules

1. Notebooks live in notebooks/, named NN_topic.ipynb (two-digit step number), one topic each.
2. Notebooks are thin: they import functions from src/. Never copy logic. If a function is missing, add it to a NEW module under src/ and import it.
3. First cell: title, purpose, inputs, outputs, runtime. Second cell: setup (see src/nbutils.py), flags RUN_HEAVY = False and RUN_API = False.
4. Heavy steps (raw CSV scan, more than 10 API requests) run only when the flag is True; otherwise load the saved artifact from data/processed and state which script and file produced it. Light API calls (up to 10 requests) are allowed inside try/except.
5. Every table that can be recomputed from data/processed in under 60 seconds is recomputed in the notebook and compared with the saved artifact.
6. Each notebook ends with: a Checks cell (Checker from src/nbutils.py), a Findings cell built with f-strings from computed variables (Markdown via IPython.display), and a Limitations markdown cell.
7. Plots: matplotlib with title and axis labels; saved to report/figures/NN_name.png (dpi 150) and shown inline. Each notebook under 2 MB.
8. No absolute paths in outputs (print paths relative to the repo root); never print more than 40 rows (use head()).
9. Execute with: python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 <notebook>, then run python tests/check_notebooks.py.
10. Never fit scalers or encoders before the train/test split; supervised models use time-based splits.
11. If a check fails, do not change the expected value: show expected and actual and report it.
12. Notebooks never write into data/ or notebooks/. Every read and write uses ROOT-based paths; figures go only to ROOT / 'report/figures'.

