"""
Unit tests for tests/check_notebooks.py
Verifies that check_notebook correctly flags a failing notebook with status FAIL.
"""

import tempfile
from pathlib import Path
import nbformat as nbf
from tests.check_notebooks import check_notebook


def test_check_notebooks_fails_on_failing_checks():
    """Build a minimal temp .ipynb whose last code cell prints 'CHECKS: 1 passed, 1 failed' and assert status FAIL."""
    nb = nbf.v4.new_notebook()
    cell = nbf.v4.new_code_cell(
        source="print('CHECKS: 1 passed, 1 failed')",
        execution_count=1,
        outputs=[
            nbf.v4.new_output(
                output_type="stream",
                name="stdout",
                text="CHECKS: 1 passed, 1 failed\n",
            )
        ],
    )
    nb.cells = [cell]

    with tempfile.TemporaryDirectory() as tmpdir:
        nb_path = Path(tmpdir) / "failing_sample.ipynb"
        with open(nb_path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)

        res = check_notebook(nb_path)
        assert res["status"] == "FAIL"
        assert res["n_failed"] == 1
        assert "CHECKS: 1 passed, 1 failed" in res["checks_line"]

