"""
Stage s06: amplitude likelihood checks. Prototype Step 6.
Run in parts (each fits in a few minutes):
    python stages/s06_likelihood.py closedform
    python stages/s06_likelihood.py gauss_D      # 800 Gaussian draws, D_true = D_inj
    python stages/s06_likelihood.py gauss_0      # 800 Gaussian draws, D_true = 0
    python stages/s06_likelihood.py mocks_0      # 300 Poisson mocks, no dipole
    python stages/s06_likelihood.py isotropic    # bias with equal vs actual errors
    python stages/s06_likelihood.py summary

Checks
  * closed form: with covariance sigma^2 I, L(A) ~ exp[-(A^2+|d|^2)/2s^2] sinh(x)/x
  * grid resolution (16, 1500) agrees with (32, 6000)
  * zero-dipole Poisson mocks agree with the Gaussian study (error model is right)
Recorded: bias of naive |d|, peak, mean, median of L(A); the bias of the
peak with equal vs actual errors (corrected explanation, see AI_log).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import config
from dipole import (unit_vector, cos_theta_map, ecliptic_latitude_map, load_mask, load_inputs,
                    expected_counts, draw_mock, fit_dipole_lsq_mult, amplitude_likelihood)
from dipole.checks import StageRecord, read_stage

part = sys.argv[1] if len(sys.argv) > 1 else "summary"
I = load_inputs()
D = I["D_kin_from_published"]; dhat = unit_vector(); cth = cos_theta_map(dhat)
ae = np.abs(ecliptic_latitude_map()); mask = load_mask("s22")
kw = dict(N0=I["ecl_intercept_count"], s=I["ecl_frac_slope_per_deg"], D=D)
cov = fit_dipole_lsq_mult(expected_counts(cth, ae, mask, **kw), mask, ae)["cov_d"]
Lchol = np.linalg.cholesky(cov)
OUT = config.stage_dir("s06_likelihood")
NAMES = ["naive |d|", "peak of L", "mean of L", "median of L"]


def summaries(dv, cv):
    r = amplitude_likelihood(dv, cv)
    return (np.linalg.norm(dv), r["peak"], r["mean"], r["median"])


if part == "closedform":
    rec = StageRecord("s06_likelihood", part)
    rec.number("sigma_per_component", np.sqrt(np.diag(cov)))
    s0 = 0.0015; worst = 0
    for dmag in (0.0, 0.003, 0.00727, 0.015):
        r = amplitude_likelihood(dmag * dhat, s0**2 * np.eye(3))
        A = r["A"]; x = A * dmag / s0**2
        La = np.exp(-(A**2 + dmag**2) / (2 * s0**2)) * np.where(x > 1e-8, np.sinh(x) / np.maximum(x, 1e-300), 1.0)
        La /= La.max(); worst = max(worst, np.max(np.abs(r["L"] - La)))
    rec.check("grid L(A) matches the closed form", worst < 1e-3, f"max difference {worst:.1e}")
    rng = np.random.default_rng(0); worst_g = 0
    for Dt in (D, 0.0):
        dv = Dt * dhat + Lchol @ rng.standard_normal(3)
        a, b = amplitude_likelihood(dv, cov), amplitude_likelihood(dv, cov, nA=6000, nside_dir=32)
        if Dt:
            worst_g = max(worst_g, abs(a["peak"] / b["peak"] - 1), abs(a["mean"] / b["mean"] - 1), abs(a["median"] / b["median"] - 1))
    rec.check("grid (16, 1500) agrees with (32, 6000)", worst_g < 1e-3, f"largest relative difference {worst_g:.1e}")
    rec.write()

elif part in ("gauss_D", "gauss_0"):
    Dt = D if part == "gauss_D" else 0.0
    rng = np.random.default_rng(61 if Dt else 60)
    rows = np.array([summaries(Dt * dhat + Lchol @ rng.standard_normal(3), cov) for _ in range(800)])
    np.save(OUT / f"{part}.npy", rows); print(f"  saved {len(rows)} draws")

elif part == "mocks_0":
    rng = np.random.default_rng(62)
    lam0 = expected_counts(cth, ae, mask, dipole=False, **kw)
    rows = []
    for _ in range(300):
        f = fit_dipole_lsq_mult(draw_mock(lam0, rng), mask, ae)
        rows.append(summaries(f["d"], f["cov_d"]))
    np.save(OUT / "mocks_0.npy", np.array(rows)); print("  saved 300 mocks")

elif part == "isotropic":
    rec = StageRecord("s06_likelihood", part)
    rng = np.random.default_rng(5)
    iso = (np.trace(cov) / 3) * np.eye(3)
    for lab, cv in [("equal errors", iso), ("actual errors", cov)]:
        Lc = np.linalg.cholesky(cv)
        pk = np.array([amplitude_likelihood(D * dhat + Lc @ rng.standard_normal(3), cv, nside_dir=8)["peak"]
                       for _ in range(2000)])
        b, e = pk.mean() / D - 1, pk.std() / np.sqrt(pk.size) / D
        rec.number(f"peak_bias_{lab.replace(' ', '_')}", dict(bias=b, error=e))
        print(f"  {lab}: peak bias {100*b:+.2f}% +/- {100*e:.2f}%")
    rec.note("The peak's residual bias comes from unequal errors between directions (z is ~1.7x better "
             "measured than x, y because of the Galactic mask), not from a sigma^2/(2D) term.")
    rec.write()

elif part == "summary":
    rec = StageRecord("s06_likelihood", "summary")
    res = {}
    for nm, Dt in [("gauss_D", D), ("gauss_0", 0.0), ("mocks_0", 0.0)]:
        rows = np.load(OUT / f"{nm}.npy"); n = len(rows); res[nm] = rows
        print(f"  {nm}: D_true = {Dt:.5f}, {n} realizations")
        for j, lab in enumerate(NAMES):
            m, se = rows[:, j].mean(), rows[:, j].std(ddof=1) / np.sqrt(n)
            rec.number(f"{nm}_{lab}", dict(mean=m, se=se, rel_bias=(m / Dt - 1) if Dt else None))
            print(f"     {lab:12s}: {m:.5f} +/- {se:.5f}" + (f" ({100*(m/Dt-1):+.1f}%)" if Dt else ""))
    for j, lab in enumerate(NAMES):
        a, b = res["mocks_0"][:, j], res["gauss_0"][:, j]
        z = (a.mean() - b.mean()) / np.hypot(a.std(ddof=1) / np.sqrt(a.size), b.std(ddof=1) / np.sqrt(b.size))
        rec.check(f"zero-dipole mocks agree with Gaussian study ({lab})", abs(z) < 3, f"{z:+.2f} SE")
    rec.write()
