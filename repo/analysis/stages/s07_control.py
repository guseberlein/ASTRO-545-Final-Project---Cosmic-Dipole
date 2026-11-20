"""
Stage s07: control run (dipole + Poisson noise only, no ecliptic pattern,
no template). Prototype Step 7 (added to the original plan).
Run in parts:
    python stages/s07_control.py mocks 0     # mocks 0-249
    python stages/s07_control.py mocks 1     # mocks 250-499
    python stages/s07_control.py predict     # expected average peak of L(A)
    python stages/s07_control.py summary

Checks
  1. average fitted dipole VECTOR = injected vector (each component within 3 SE)
  2. scatter of each component = error reported by the fit (within 3/sqrt(2(N-1)))
  3. average PEAK of L(A) = prediction for this noise level (within 3 SE)
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, healpy as hp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import config
from dipole import (unit_vector, cos_theta_map, ecliptic_latitude_map, load_mask, load_inputs,
                    expected_counts, draw_mock, fit_dipole_lsq, amplitude_likelihood)
from dipole.checks import StageRecord

part = sys.argv[1] if len(sys.argv) > 1 else "summary"
I = load_inputs()
D = I["D_kin_from_published"]; dhat = unit_vector(); d_true = D * dhat
mask = load_mask("s22"); cth = cos_theta_map(dhat); ae = np.abs(ecliptic_latitude_map())
lam = expected_counts(cth, ae, mask, N0=I["ecl_intercept_count"], s=I["ecl_frac_slope_per_deg"], D=D,
                      ecliptic=False)
NMOCK, CHUNK = 500, 250
OUT = config.stage_dir("s07_control")

if part == "mocks":
    c = int(sys.argv[2]); rows = []
    for i in range(c * CHUNK, (c + 1) * CHUNK):
        rng = np.random.default_rng(7_000_000 + i)
        f = fit_dipole_lsq(draw_mock(lam, rng), mask)
        r = amplitude_likelihood(f["d"], f["cov"][:3, :3])
        rows.append(np.r_[f["d"], f["sigma"][:3], f["amp"], r["peak"], r["lo"], r["hi"]])
    np.save(OUT / f"mocks_{c}.npy", np.array(rows)); print(f"  saved mocks {c*CHUNK}-{(c+1)*CHUNK-1}")

elif part == "predict":
    cv = fit_dipole_lsq(lam, mask)["cov"][:3, :3]
    Lc = np.linalg.cholesky(cv); rng = np.random.default_rng(77)
    pk = np.array([amplitude_likelihood(d_true + Lc @ rng.standard_normal(3), cv, nside_dir=8)["peak"]
                   for _ in range(3000)])
    np.save(OUT / "predict.npy", pk)
    print(f"  predicted average peak {pk.mean():.6f} +/- {pk.std(ddof=1)/np.sqrt(pk.size):.6f}")

elif part == "summary":
    rec = StageRecord("s07_control", "summary")
    R = np.vstack([np.load(OUT / f"mocks_{c}.npy") for c in range(NMOCK // CHUNK)])
    d, sig_rep, amp, peak = R[:, :3], R[:, 3:6], R[:, 6], R[:, 7]; N = len(R)
    for j, ax_ in enumerate("xyz"):
        m, sd = d[:, j].mean(), d[:, j].std(ddof=1); z = (m - d_true[j]) / (sd / np.sqrt(N))
        rec.check(f"1. average d_{ax_} = injected", abs(z) < 3, f"{m:+.6f} vs {d_true[j]:+.6f} ({z:+.2f} SE)")
    for j, ax_ in enumerate("xyz"):
        ratio = d[:, j].std(ddof=1) / sig_rep[:, j].mean(); tol = 1 / np.sqrt(2 * (N - 1))
        rec.check(f"2. scatter/reported error ({ax_})", abs(ratio - 1) < 3 * tol, f"{ratio:.3f} (1 +/- {tol:.3f})")
    pk = np.load(OUT / "predict.npy"); pred, pse = pk.mean(), pk.std(ddof=1) / np.sqrt(pk.size)
    m, se = peak.mean(), peak.std(ddof=1) / np.sqrt(N); z3 = (m - pred) / np.hypot(se, pse)
    rec.check("3. average peak of L(A) = prediction", abs(z3) < 3,
              f"{m:.6f} +/- {se:.6f} vs {pred:.6f} +/- {pse:.6f} ({z3:+.2f} SE)")
    dm = d.mean(axis=0)
    rec.number("average_peak_over_D", m / D); rec.number("predicted_peak_over_D", pred / D)
    rec.number("naive_length_over_D", amp.mean() / D)
    rec.number("average_vector_angle_deg", float(np.degrees(np.arccos(dm @ dhat / np.linalg.norm(dm)))))
    ll, bb = hp.vec2ang(d / np.linalg.norm(d, axis=1)[:, None], lonlat=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for j, ax_ in enumerate("xyz"):
        zz = (d[:, j] - d_true[j]) / sig_rep[:, j]
        ax[0].hist(zz, bins=30, histtype="step", density=True, label=f"{ax_}: std {zz.std(ddof=1):.3f}")
    xx = np.linspace(-4, 4, 200); ax[0].plot(xx, np.exp(-xx**2 / 2) / np.sqrt(2 * np.pi), "k--")
    ax[0].set_xlabel("(fitted - injected) / reported error"); ax[0].legend(fontsize=8)
    ax[1].scatter(np.where(ll > 180, ll - 360, ll), bb, s=4, alpha=0.5)
    ax[1].plot(config.L_CMB_DEG - 360, config.B_CMB_DEG, "r*", ms=15); ax[1].set_xlabel("l (wrapped)"); ax[1].set_ylabel("b")
    plt.tight_layout(); plt.savefig(OUT / "control.png", dpi=110); plt.close()
    rec.write()
