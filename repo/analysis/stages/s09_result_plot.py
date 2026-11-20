"""
Stage s09: the prototype's output plot and headline numbers. Prototype Step 9.

Checks
  1. plotted numbers are computed from the s08 results (and agree with them)
  2. reproducibility: regenerating 5 mocks from their stored seeds and
     refitting reproduces the stored s08 results exactly
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import config
from dipole import (unit_vector, cos_theta_map, ecliptic_latitude_map, load_mask, load_inputs,
                    expected_counts, draw_mock, fit_dipole_lsq_mult)
from dipole.checks import StageRecord, read_stage

rec = StageRecord("s09_result_plot")
I = load_inputs(); D = I["D_kin_from_published"]
S8 = config.RESULTS / "s08_main"
load = lambda k: np.vstack([np.load(S8 / f"{k}_{c}.npy") for c in range(4)])
A, B = load("A"), load("B")
pa, pb = A[:, 7] / D, B[:, 7] / D; N = len(pa)
mA, sA, mB, sB = pa.mean(), pa.std(ddof=1) / np.sqrt(N), pb.mean(), pb.std(ddof=1) / np.sqrt(N)
dif = pb - pa; mD, sD = dif.mean(), dif.std(ddof=1) / np.sqrt(N)
for kk, v in [("A_over_D", (mA, sA)), ("B_over_D", (mB, sB)), ("paired_B_minus_A_over_D", (mD, sD))]:
    rec.number(kk, dict(value=v[0], se=v[1]))
s8 = read_stage("s08_main", "summary")["numbers"]
rec.check("1. numbers agree with s08 summary",
          abs(mA - s8["A_average_peak_over_D"]["value"]) < 1e-12 and abs(mD - s8["paired_B_minus_A_over_D"]["value"]) < 1e-12,
          f"A {mA:.4f}+/-{sA:.4f}, B {mB:.4f}+/-{sB:.4f}, B-A {mD:+.4f}+/-{sD:.4f}")

mask = load_mask("s22"); ae = np.abs(ecliptic_latitude_map())
lam = expected_counts(cos_theta_map(unit_vector()), ae, mask, N0=I["ecl_intercept_count"],
                      s=I["ecl_frac_slope_per_deg"], D=D)
worst = max(np.max(np.abs(fit_dipole_lsq_mult(draw_mock(lam, np.random.default_rng(8_000_000 + i)), mask, ae)["d"] - A[i, :3]))
            for i in range(5))
rec.check("2. reproducible from stored seeds", worst < 1e-12, f"largest difference {worst:.1e}")

C_A, C_B, INK, INK2, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb"
fig, ax = plt.subplots(figsize=(8.5, 5), facecolor=SURF); ax.set_facecolor(SURF)
bins = np.linspace(0.3, 1.9, 41)
for p, c, lab, m, s in [(pa, C_A, "A: with ecliptic template", mA, sA), (pb, C_B, "B: without ecliptic template", mB, sB)]:
    ax.hist(p, bins=bins, color=c, alpha=0.28, edgecolor="none")
    ax.hist(p, bins=bins, histtype="step", color=c, lw=2, label=f"{lab}\n    mean {m:.3f} ± {s:.3f} (dashed line)")
    ax.axvline(m, color=c, lw=2, ls="--")
ax.axvline(1.0, color=INK, lw=1.5); ax.set_ylim(0, 62)
ax.text(0.99, 0.985, "injected\namplitude", transform=ax.get_xaxis_transform(), va="top", ha="right", fontsize=10)
ax.set_xlabel("Recovered dipole amplitude ÷ injected amplitude  (peak of L(A))"); ax.set_ylabel("Number of mocks (of 500)")
ax.set_title(f"Leaving out the ecliptic template inflates the recovered dipole by {100*mD:.1f}% ± {100*sD:.1f}%", loc="left", fontsize=12)
ax.legend(frameon=False, loc="upper right"); ax.grid(axis="y", color="#e4e3df"); ax.set_axisbelow(True)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
plt.tight_layout(); plt.savefig(rec.dir / "result.png", dpi=150, facecolor=SURF); plt.close()
rec.write()
