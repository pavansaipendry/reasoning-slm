"""Sandboxed code-execution scorer — shared by code eval and the GRPO code reward.

Runs model-generated code plus unit tests in a separate Python process with a wall-clock
timeout, so a verifiable pass/fail signal can be computed. This is a *lightweight* sandbox
(subprocess isolation + timeout + isolated interpreter), appropriate for evaluating a
model's own generations or trusted test suites — not a hardened jail for adversarial code.

Public API:
    extract_code(completion) -> str         # pull code out of a model completion
    run_unit_tests(code, tests) -> float    # fraction of test groups that pass (0..1)
    code_reward(completion, tests) -> float  # verifiable reward for GRPO
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile

# code fenced as ```python ... ``` (or ```), else fall back to an <answer> block.
_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def extract_code(completion: str) -> str:
    m = _FENCE.findall(completion)
    if m:
        return m[-1].strip()
    m = _ANSWER.findall(completion)
    if m:
        return m[-1].strip()
    return completion.strip()


def _run_one(source: str, timeout_s: float) -> bool:
    """Run a self-contained script; True iff it exits 0 within the timeout."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=True) as f:
        f.write(source)
        f.flush()
        try:
            proc = subprocess.run(
                [sys.executable, "-I", f.name],          # -I: isolated, ignore env/site
                capture_output=True, text=True, timeout=timeout_s,
            )
            return proc.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False


def run_unit_tests(code: str, tests, timeout_s: float = 5.0) -> float:
    """Fraction of test groups that pass.

    `tests` may be a single string (one group) or a list of strings (each scored
    independently, so partial credit is possible). Each test group is appended to
    `code` and executed; passing == exits cleanly (asserts didn't fire)."""
    groups = [tests] if isinstance(tests, str) else list(tests)
    if not groups:
        return 0.0
    passed = sum(_run_one(code + "\n\n" + t, timeout_s) for t in groups)
    return passed / len(groups)


def code_reward(completion: str, tests, timeout_s: float = 5.0) -> float:
    """Verifiable reward for GRPO on code tasks: pass-rate of the extracted code."""
    return run_unit_tests(extract_code(completion), tests, timeout_s)


if __name__ == "__main__":
    # tiny self-test
    good = "```python\ndef add(a, b):\n    return a + b\n```"
    bad = "```python\ndef add(a, b):\n    return a - b\n```"
    t = ["assert add(2, 3) == 5", "assert add(0, 0) == 0"]
    print("good:", run_unit_tests(extract_code(good), t))   # 1.0
    print("bad :", run_unit_tests(extract_code(bad), t))    # 0.5 (only add(0,0) passes)
