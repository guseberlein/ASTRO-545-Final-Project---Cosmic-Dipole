"""
Collect every PASS/FAIL check from results/*/summary*.json into one table,
results/ALL_CHECKS.md, and print a one-line total per stage.
"""
import sys, pathlib, json
ROOT = pathlib.Path(__file__).resolve().parent
rows, lines = [], ["# All checks", "", "| Stage | Part | Check | Result | Detail |", "|---|---|---|---|---|"]
for f in sorted((ROOT / "results").glob("*/summary*.json")):
    s = json.loads(f.read_text())
    n = len(s["checks"]); nf = sum(not c["passed"] for c in s["checks"])
    rows.append((s["stage"], s["part"], n, nf))
    for c in s["checks"]:
        det = str(c["detail"]).replace("|", "/")
        lines.append(f"| {s['stage']} | {s['part'] or ''} | {c['name']} | {'PASS' if c['passed'] else 'FAIL'} | {det} |")
fails = []
for f in sorted((ROOT / "results").glob("*/summary*.json")):
    s = json.loads(f.read_text())
    if any(not c["passed"] for c in s["checks"]):
        fails.append(f"- **{s['stage']} {s['part'] or ''}**: " + " ".join(s["notes"]))
if fails:
    lines += ["", "## Failed checks, explained", ""] + fails
v = ROOT / "results" / "verification_against_prototype.json"
if v.exists():
    vv = json.loads(v.read_text())
    lines += ["", f"Verification against prototype: {vv['n_items']} items compared, {vv['n_failed']} failed."]
(ROOT / "results" / "ALL_CHECKS.md").write_text("\n".join(lines) + "\n")
tot = sum(r[2] for r in rows); totf = sum(r[3] for r in rows)
for st, pt, n, nf in rows:
    print(f"  {st:22s} {str(pt or ''):11s} {n:3d} checks, {nf} failed")
print(f"TOTAL: {tot} checks, {totf} failed  -> results/ALL_CHECKS.md")
