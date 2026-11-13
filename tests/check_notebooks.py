"""
Verification script for CityPulse Jupyter Notebooks.
Ensures:
- Every code cell has an execution_count
- No output has output_type 'error'
- Notebook file size is under 2 MB
- No absolute 'C:\\Users' or 'C:/Users' strings in output text
- The last code cell output contains 'CHECKS: N passed, M failed'
- Exits non-zero on structural failures or failed checks; non-zero failed checks result in a FAIL status.
"""

import sys
import re
from pathlib import Path
import nbformat

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"


def check_notebook(nb_path: Path):
    structural_errors = []
    file_size_bytes = nb_path.stat().st_size
    size_kb = file_size_bytes / 1024.0

    if file_size_bytes >= 2 * 1024 * 1024:
        structural_errors.append("File size >= 2 MB")

    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
    except Exception as e:
        return {
            "notebook": nb_path.name,
            "cells": 0,
            "size_kb": f"{size_kb:.1f}",
            "checks_line": "ERROR",
            "status": f"FAIL (Could not parse notebook: {e})",
            "has_structural_error": True,
        }

    total_cells = len(nb.cells)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]

    for idx, cell in enumerate(code_cells, 1):
        if cell.execution_count is None:
            structural_errors.append(f"Code cell {idx} unexecuted (execution_count is None)")

        for out_idx, out in enumerate(cell.get("outputs", []), 1):
            if out.get("output_type") == "error":
                ename = out.get("ename", "Error")
                evalue = out.get("evalue", "")
                structural_errors.append(f"Cell {idx} error output: {ename}: {evalue}")

            # Collect text to check for forbidden paths
            texts = []
            if "text" in out:
                texts.append(out["text"])
            if "data" in out and isinstance(out["data"], dict):
                for mime_val in out["data"].values():
                    if isinstance(mime_val, str):
                        texts.append(mime_val)
                    elif isinstance(mime_val, list):
                        texts.append("".join(str(x) for x in mime_val))
            if "traceback" in out and isinstance(out["traceback"], list):
                texts.append("".join(out["traceback"]))

            combined_out = "\n".join(texts)
            if re.search(r"(?i)[a-z]:[\\/]+users", combined_out):
                structural_errors.append(f"Cell {idx} output contains forbidden user path matching (?i)[a-z]:[\\/]+users")

    checks_line = "MISSING"
    n_failed = 0
    if not code_cells:
        structural_errors.append("No code cells found in notebook")
    else:
        last_code_cell = code_cells[-1]
        last_out_texts = []
        for out in last_code_cell.get("outputs", []):
            if "text" in out:
                last_out_texts.append(out["text"])
            if "data" in out and isinstance(out["data"], dict):
                for mime_val in out["data"].values():
                    if isinstance(mime_val, str):
                        last_out_texts.append(mime_val)
                    elif isinstance(mime_val, list):
                        last_out_texts.append("".join(str(x) for x in mime_val))
        combined_last = "\n".join(last_out_texts)
        m = re.search(r"CHECKS:\s*(\d+)\s+passed,\s*(\d+)\s+failed", combined_last)
        if m:
            checks_line = m.group(0)
            n_passed = int(m.group(1))
            n_failed = int(m.group(2))
        else:
            structural_errors.append("Last code cell output missing line matching 'CHECKS: N passed, M failed'")

    if structural_errors:
        status = "FAIL: " + "; ".join(structural_errors)
        has_structural_error = True
    elif n_failed > 0:
        status = "FAIL"
        has_structural_error = False
    else:
        status = "PASS"
        has_structural_error = False

    return {
        "notebook": nb_path.name,
        "cells": total_cells,
        "size_kb": f"{size_kb:.1f}",
        "checks_line": checks_line,
        "status": status,
        "has_structural_error": has_structural_error,
        "n_failed": n_failed,
    }


def main():
    # Check for stray entries in notebooks/
    stray_entries = [
        e.name for e in NOTEBOOKS_DIR.iterdir()
        if not (e.name.endswith(".ipynb") or e.name in (".ipynb_checkpoints", ".gitkeep"))
    ]
    stray_failure = False
    if stray_entries:
        print(f"FAIL: Stray entries found in notebooks/ (only *.ipynb files and .ipynb_checkpoints allowed): {', '.join(stray_entries)}")
        stray_failure = True

    nb_files = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))
    if not nb_files:
        print("No notebooks found in notebooks/")
        sys.exit(1)

    results = [check_notebook(p) for p in nb_files]

    # Print table
    col_nb = max(len("Notebook"), max(len(r["notebook"]) for r in results))
    col_cells = max(len("Cells"), max(len(str(r["cells"])) for r in results))
    col_size = max(len("Size (KB)"), max(len(r["size_kb"]) for r in results))
    col_chk = max(len("Checks Line"), max(len(r["checks_line"]) for r in results))
    col_st = max(len("Status"), max(len(r["status"]) for r in results))

    header = f"| {'Notebook'.ljust(col_nb)} | {'Cells'.rjust(col_cells)} | {'Size (KB)'.rjust(col_size)} | {'Checks Line'.ljust(col_chk)} | {'Status'.ljust(col_st)} |"
    sep = f"| {'-' * col_nb} | {'-' * col_cells} | {'-' * col_size} | {'-' * col_chk} | {'-' * col_st} |"
    print(header)
    print(sep)
    for r in results:
        row = f"| {r['notebook'].ljust(col_nb)} | {str(r['cells']).rjust(col_cells)} | {r['size_kb'].rjust(col_size)} | {r['checks_line'].ljust(col_chk)} | {r['status'].ljust(col_st)} |"
        print(row)

    has_any_failure = any(r["has_structural_error"] or r["n_failed"] > 0 for r in results) or stray_failure
    if has_any_failure:
        print("\nResult: FAIL")
        sys.exit(1)
    else:
        print("\nResult: All notebooks passed structural checks.")
        sys.exit(0)


if __name__ == "__main__":
    main()

