"""
Stage s08: main run. Prototype Step 8.
500 mocks with the injected dipole AND the planted multiplicative ecliptic
pattern; each fitted three ways on identical noise:
  A  with ecliptic template, multiplicative estimator (main choice)
  B  without template
  C  G&H additive ecliptic template (for comparison)
Run in parts:
    python stages/s08_main.py mocks 0 ... mocks 3   (125 mocks each)
    python stages/s08_main.py predict A|B|C
    python stages/s08_main.py summary

Checks (for each fit, against what that fit returns on the NOISELESS mock;
for A that is exactly the injected dipole):
  1. average vector = expected (each component within 3 SE)
  2. scatter = reported error
  3. average peak of L(A) = prediction for that fit's noise level
  4. (A only) average fitted ecliptic slope = planted slope
Result: paired difference B - A in recovered amplitude, and direction change.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import config
from dipole import (unit_vector, cos_theta_map, ecliptic_latitude_map, load_mask, load_inputs,
                    expected_counts, draw_mock, fit_dipole_lsq, fit_dipole_lsq_mult, amplitude_likelihood)
from dipole.checks import StageRecord

part = sys.argv[1] if len(sys.argv) > 1 else "summary"
I = load_inputs()
D = I["D_kin_from_published"]; s_planted = I["ecl_frac_slope_per_deg"]
dhat = unit_vector(); mask = load_mask("s22")
cth = cos_theta_map(dhat); ae = np.abs(ecliptic_latitude_map())
lam = expected_counts(cth, ae, mask, N0=I["ecl_intercept_count"], s=s_planted, D=D)
NMOCK, CHUNK = 500, 125
OUT = config.stage_dir("s08_main")
ecl_t = {"ecl": ae}


def fitA(m): f = fit_dipole_lsq_mult(m, mask, ae); return f["d"], f["cov_d"], f["s"]
def fitB(m): f = fit_dipole_lsq(m, mask); return f["d"], f["cov"][:3, :3], np.nan
def fitC(m): f = fit_dipole_lsq(m, mask, ecl_t); return f["d"], f["cov"][:3, :3], f["k"]["ecl"]
FITS = {"A": fitA, "B": fitB, "C": fitC}
LABEL = {"A": "with ecliptic template (multiplicative)", "B": "without ecliptic template",
         "C": "G&H additive ecliptic template"}

if part == "mocks":
    c = int(sys.argv[2]); out = {k: [] for k in FITS}
    for i in range(c * CHUNK, (c + 1) * CHUNK):
        m = draw_mock(lam, np.random.default_rng(8_000_000 + i))
        for k, fn in FITS.items():
            d, cv, extra = fn(m)
            r = amplitude_likelihood(d, cv)
            out[k].append(np.r_[d, np.sqrt(np.diag(cv)), np.linalg.norm(d), r["peak"], extra])
    for k in FITS:
        np.save(OUT / f"{k}_{c}.npy", np.array(out[k]))
    print(f"  saved mocks {c*CHUNK}-{(c+1)*CHUNK-1}")

elif part == "predict":
    k = sys.argv[2]
    d0, cv, _ = FITS[k](lam)
    Lc = np.linalg.cholesky(cv); rng = np.random.default_rng(88)
    pk = np.array([amplitude_likelihood(d0 + Lc @ rng.standard_normal(3), cv, nside_dir=8)["peak"]
                   for _ in range(1500)])
    np.save(OUT / f"predict_{k}.npy", np.r_[d0, pk.mean(), pk.std(ddof=1) / np.sqrt(pk.size)])
    print(f"  {k}: noiseless |d| = {np.linalg.norm(d0)/D:.4f} x D_inj; predicted peak {pk.mean():.6f}")

elif part == "summary":
    rec = StageRecord("s08_main", "summary")
    res = {}
    for k in FITS:
        R = np.vstack([np.load(OUT / f"{k}_{c}.npy") for c in range(NMOCK // CHUNK)]); res[k] = R
        pred = np.load(OUT / f"predict_{k}.npy"); d0 = pred[:3]
        N = len(R); d, sg, peak, extra = R[:, :3], R[:, 3:6], R[:, 7], R[:, 8]
        print(f"  --- {k}: {LABEL[k]}; expected average |d| = {np.linalg.norm(d0)/D:.4f} x D_inj")
        for j, ax_ in enumerate("xyz"):
            m, sd = d[:, j].mean(), d[:, j].std(ddof=1); z = (m - d0[j]) / (sd / np.sqrt(N))
            ratio = sd / sg[:, j].mean()
            rec.check(f"{k}1. average d_{ax_} = expected", abs(z) < 3, f"{z:+.2f} SE")
            rec.check(f"{k}2. scatter/reported error ({ax_})", abs(ratio - 1) < 3 / np.sqrt(2 * (N - 1)), f"{ratio:.3f}")
        mp, se = peak.mean(), peak.std(ddof=1) / np.sqrt(N); z3 = (mp - pred[3]) / np.hypot(se, pred[4])
        rec.check(f"{k}3. average peak = prediction", abs(z3) < 3, f"{mp:.6f} vs {pred[3]:.6f} ({z3:+.2f} SE)")
        if k == "A":
            zs = (extra.mean() - s_planted) / (extra.std(ddof=1) / np.sqrt(N))
            rec.check("A4. fitted ecliptic slope = planted", abs(zs) < 3, f"{extra.mean():.4e} vs {s_planted:.4e} ({zs:+.2f} SE)")
        dm = d.mean(axis=0)
        rec.number(f"{k}_average_peak_over_D", dict(value=mp / D, se=se / D))
        rec.number(f"{k}_noiseless_over_D", np.linalg.norm(d0) / D)
        rec.number(f"{k}_average_vector_angle_deg", float(np.degrees(np.arccos(dm @ dhat / np.linalg.norm(dm)))))
    dif = (res["B"][:, 7] - res["A"][:, 7]) / D
    ang = np.degrees(np.arccos(np.sum(res["A"][:, :3] * res["B"][:, :3], axis=1) / (res["A"][:, 6] * res["B"][:, 6])))
    rec.number("paired_B_minus_A_over_D", dict(value=dif.mean(), se=dif.std(ddof=1) / np.sqrt(dif.size)))
    rec.number("median_A_B_angle_deg", float(np.median(ang)))
    bias_vec = res["B"][:, :3].mean(0) - res["A"][:, :3].mean(0)
    rec.number("B_minus_A_mean_vector", bias_vec); rec.number("B_minus_A_mean_vector_length", np.linalg.norm(bias_vec))
    print(f"  RESULT: B - A = {dif.mean():+.4f} +/- {dif.std(ddof=1)/np.sqrt(dif.size):.4f} x D_inj; "
          f"median angle {np.median(ang):.1f} deg")
    rec.write()
