"""Check the parallel results against the serial ones and the expected accuracy.

usage: python3 analysis/validate.py [results_dir]      (default: results)
  gpu_validate.csv : serial + MPI+CUDA runs with a fixed iteration count
  cpu_validate.csv : serial + MPI (CPU) runs with a fixed iteration count
  accuracy.csv     : converged runs for several grid sizes
Exit code 0 = everything passed, 1 = a check failed (or nothing could be compared).
"""
import csv
import sys
from collections import defaultdict

RES = sys.argv[1] if len(sys.argv) > 1 else "results"

TOL = 1e-9  # relative tolerance for sum_u and l2_err (the summation order differs)


def rows(path):
    try:
        with open(path) as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        print(f"(not available yet: {path})")
        return []


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


ok = True
compared = passed = 0
for name in ("gpu_validate", "cpu_validate"):
    groups = defaultdict(list)
    for r in rows(f"{RES}/{name}.csv"):
        groups[(r["dim"], r["n"], r["iters"])].append(r)
    if not groups:
        continue
    print(f"== {name}: parallel vs serial (fixed iteration count) ==")
    for key, rs in sorted(groups.items()):
        ref = next((r for r in rs if r["version"] == "serial"), None)
        if ref is None:
            continue
        for r in rs:
            if r is ref:
                continue
            e1 = rel(float(r["sum_u"]), float(ref["sum_u"]))
            e2 = rel(float(r["l2_err"]), float(ref["l2_err"]))
            good = e1 < TOL and e2 < TOL
            compared += 1
            passed += good
            ok &= good
            print(f"{r['version']} dim={key[0]} n={key[1]} ranks={r['ranks']} "
                  f"grid=({r['px']},{r['py']},{r['pz']}) cudaaware={r['cudaaware']} "
                  f"overlap={r['overlap']} prio={r['prio']}  "
                  f"rel(sum_u)={e1:.1e} rel(l2)={e2:.1e}  {'PASS' if good else 'FAIL'}")
if compared == 0:
    print("NO parallel-vs-serial comparison possible (no parallel results)")
    ok = False
else:
    print(f"\n{passed} of {compared} parallel runs reproduce the serial result")

print("\n== error vs exact solution: expect factor ~4 per grid refinement ==")
acc = defaultdict(list)
for r in rows(f"{RES}/accuracy.csv"):
    acc[r["dim"]].append((int(r["n"]), float(r["l2_err"])))
for dim, pts in sorted(acc.items()):
    pts = sorted(set(pts))
    for (n0, e0), (n1, e1) in zip(pts, pts[1:]):
        ratio = e0 / e1
        good = 3.5 < ratio < 4.5
        ok &= good
        print(f"dim={dim} n {n0}->{n1}: l2 {e0:.3e} -> {e1:.3e}  ratio={ratio:.2f}  "
              f"{'PASS' if good else 'FAIL'}")

print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
