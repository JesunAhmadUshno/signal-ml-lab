"""Compare JSON results with a stated float tolerance.

Why: NumPy's float output can differ in the last digits between CPUs
(SIMD code paths), even under the same pinned version. CI run
42548252398 failed on main for exactly that: testset.json differed at
about 1e-16 relative. So regenerated results are checked like this:
strings, booleans, integers and structure exactly; floats within
REL = 1e-9 relative (ABS = 1e-12 near zero). The dashboard's own
build-time check (site/src/signal.check.mjs) uses 1e-12 on its side.

Usage (CI): python tests/json_close.py FILE...  compares each working
file with its committed version (git show HEAD:FILE); exit 1 on drift.
"""
import json
import math
import subprocess
import sys

REL = 1e-9
ABS = 1e-12


def first_difference(a, b, path="$"):
    """Return the path of the first value that differs beyond tolerance, or None."""
    if isinstance(a, float) or isinstance(b, float):
        ok = (isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool)
              and not isinstance(b, bool) and math.isclose(a, b, rel_tol=REL, abs_tol=ABS))
        return None if ok else path
    if type(a) is not type(b):
        return path
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return path + " (keys)"
        for k in a:
            d = first_difference(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return path + " (length)"
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_difference(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    return None if a == b else path


if __name__ == "__main__":
    bad = 0
    for name in sys.argv[1:]:
        with open(name) as f:
            now = json.load(f)
        committed = json.loads(subprocess.run(["git", "show", "HEAD:" + name], check=True,
                                              capture_output=True, text=True).stdout)
        d = first_difference(now, committed)
        print(f"{name}: {'ok' if d is None else 'DRIFT at ' + d}")
        bad += d is not None
    sys.exit(1 if bad else 0)
