"""
CityPulse - Notebook Utilities
==============================
Helper functions for CityPulse thin notebooks:
- find_root(): Locate repository root containing environment.yml.
- md(text): Display Markdown in Jupyter notebooks.
- Checker: Robust verification helper for notebook checks.
"""

from pathlib import Path
from typing import Any, Optional


def find_root() -> Path:
    """Walk up from the current directory until environment.yml exists."""
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / "environment.yml").exists():
            return p.resolve()
    raise FileNotFoundError("Could not find root directory containing environment.yml")


def md(text: Any) -> None:
    """Display Markdown text in Jupyter."""
    try:
        from IPython.display import display, Markdown
        display(Markdown(str(text)))
    except ImportError:
        print(text)


class Checker:
    """
    Verification checker for notebook assertions.
    Tracks passed/failed counts and prints status without raising exceptions.
    """

    def __init__(self) -> None:
        self.passed: int = 0
        self.failed: int = 0

    def check(
        self,
        name: str,
        actual: Any,
        expected: Any,
        tol: Optional[float] = None,
    ) -> bool:
        """
        Check actual against expected value.
        If tol is provided, checks abs(actual - expected) <= tol.
        Otherwise checks exact equality.
        Never raises an exception.
        """
        try:
            if tol is not None:
                passed = bool(abs(actual - expected) <= tol)
            else:
                passed = bool(actual == expected)
        except Exception:
            passed = False

        if passed:
            self.passed += 1
            status = "PASS"
        else:
            self.failed += 1
            status = "FAIL"

        print(f"[{status}] {name}: actual={actual}, expected={expected}")
        return passed

    def summary(self) -> None:
        """Print final checks summary."""
        print(f"CHECKS: {self.passed} passed, {self.failed} failed")

