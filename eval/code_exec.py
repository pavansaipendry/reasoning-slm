"""Sandboxed code-execution scorer (Act III) — shared by eval and GRPO rewards.

SCAFFOLD. Planned: run a model-generated function against unit tests in a subprocess
with a timeout and restricted environment; return pass-rate in [0, 1]. Used both as a
HumanEval-style eval metric and as a verifiable reward signal during GRPO.
"""

from __future__ import annotations


def run_unit_tests(code: str, tests: str, timeout_s: float = 5.0) -> float:
    """Return fraction of unit tests that pass. NOT YET IMPLEMENTED."""
    raise NotImplementedError("Code-exec sandbox lands in Act III.")
