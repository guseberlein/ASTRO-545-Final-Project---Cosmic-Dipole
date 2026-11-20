"""
Stage s05: least-squares estimator checks. Prototype Step 5.

Checks (noiseless maps must be recovered exactly)
  (a) pure dipole, full sky
  (b) pure dipole, S22 footprint  (fitted d = D / (1 + D <cos theta>_footprint))
  (c) dipole + ADDITIVE ecliptic term, additive template -> dipole and coefficient exact
  (e) full-sky error per component = sqrt(3 / N_total)
  (f) agreement with healpy.fit_dipole (independent code)
  (g) real data: S21 sky map -> S21's published 0.01554 toward (238.2, 28.8)
  (h) multiplicative estimator recovers the mock-as-built exactly
Recorded (not pass/fail):
  (d) G&H additive template on the multiplicative mock: 1.0155 x D (model mismatch)
      and the no-template result 1.109 x D (preview of the main result)
  (i) the additive bias changes sign with the dipole's ecliptic latitude
      (confirms its explanation: the CMB direction lies near the ecliptic plane)
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, healpy as hp
from astropy.coordinates import SkyCoord
import astropy.units as u
import config
from dipole import (unit_vector, cos_theta_map, ecliptic_latitude_map, pixel_vectors, load_mask,
                    load_inputs, expected_counts, fit_dipole_lsq, fit_dipole_lsq_mult, NPIX)
from dipole.masks import s21_map
from dipole.checks import StageRecord

rec = StageRecord("s05_estimator")
I = load_inputs()
D = I["D_kin_from_published"]; N0, s = I["ecl_intercept_count"], I["ecl_frac_slope_per_deg"]
dhat = unit_vector(); d_true = D * dhat; cth = cos_theta_map(dhat)
ae = np.abs(ecliptic_latitude_map())
mask = load_mask("s22"); seen = ~mask; nomask = np.zeros(NPIX, bool)
ecl_t = {"ecl": ae}
angle = lambda a, b: float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))

ra = fit_dipole_lsq(60.0 * (1 + D * cth), nomask)
rec.check("(a) pure dipole, full sky: exact", np.linalg.norm(ra["d"] - d_true) < 1e-12,
          f"error {np.linalg.norm(ra['d'] - d_true):.1e}")
rb = fit_dipole_lsq(60.0 * (1 + D * cth), mask)
cm = cth[seen].mean(); expect_b = d_true / (1 + D * cm)
rec.check("(b) pure dipole, S22 footprint: exact", np.linalg.norm(rb["d"] - expect_b) < 1e-12,
          f"error {np.linalg.norm(rb['d'] - expect_b):.1e}")
m = N0 * (1 + D * cth + s * ae)
rc = fit_dipole_lsq(m, mask, ecl_t); nb = m[seen].mean()
rec.check("(c) additive ecliptic term + template: exact",
          np.linalg.norm(rc["d"] - d_true * N0 / nb) < 1e-12 and abs(rc["k"]["ecl"] / (s * N0 / nb) - 1) < 1e-10,
          f"coefficient {rc['k']['ecl']:.6e} vs {s * N0 / nb:.6e}")

lam = expected_counts(cth, ae, mask, N0=N0, s=s, D=D)
rd1, rd0 = fit_dipole_lsq(lam, mask, ecl_t), fit_dipole_lsq(lam, mask)
rec.number("d_additive_template_over_D", rd1["amp"] / D)
rec.number("d_no_template_over_D", rd0["amp"] / D)
rec.number("d_no_template_angle_deg", angle(rd0["d"], dhat))
rec.note(f"(d) G&H additive template on the multiplicative mock: {rd1['amp']/D:.4f} x D (model mismatch); "
         f"no template: {rd0['amp']/D:.4f} x D, {angle(rd0['d'], dhat):.1f} deg off")

re_ = fit_dipole_lsq(np.full(NPIX, 60.0), nomask)
expect = np.sqrt(3 / (60.0 * NPIX))
pv = pixel_vectors()
exact_grid = 1 / np.sqrt(60.0 * (pv**2).sum(axis=1))      # sum of n_i^2 over pixel centres is not exactly N/3
rec.check("(e) full-sky error = 1/sqrt(nbar * sum n_i^2) (exact for the pixel grid)",
          np.allclose(re_["sigma"][:3], exact_grid, rtol=1e-10), f"{re_['sigma'][:3]}")
rec.check("(e') full-sky error = sqrt(3/N_total) to pixelization accuracy (1e-4)",
          np.allclose(re_["sigma"][:3], expect, rtol=1e-4),
          f"relative differences {re_['sigma'][:3] / expect - 1}")
rec.note("(e) was first checked against sqrt(3/N) at 1e-6 and failed by ~1e-5: the pixel centres' "
         "average of n_z^2 is 0.3333258, not exactly 1/3. The check was corrected, not the code.")

mm = 60.0 * (1 + D * cth); mm[mask] = hp.UNSEEN
mono, dip = hp.fit_dipole(mm)
ours = np.linalg.norm(rb["d"]) / (1 + rb["C"])
rec.check("(f) agrees with healpy.fit_dipole", abs(np.linalg.norm(dip / mono) - ours) < 1e-10,
          f"{np.linalg.norm(dip/mono):.8f} vs {ours:.8f}")

s21 = s21_map(); cnt21 = np.asarray(s21["count"], float); m21 = np.asarray(s21["mask"], bool)
g_add = fit_dipole_lsq(cnt21, m21, ecl_t); g_mult = fit_dipole_lsq_mult(cnt21, m21, ae); g_none = fit_dipole_lsq(cnt21, m21)
pub = 0.01554 * unit_vector(238.2, 28.8)
sig = np.sqrt(np.trace(g_add["cov"][:3, :3]) / 3)
for lab, g in [("additive template", g_add), ("multiplicative template", g_mult), ("no template", g_none)]:
    rec.number(f"s21_real_{lab.replace(' ', '_')}", dict(D=g["amp"], l=g["l"], b=g["b"]))
rec.number("s21_real_error_per_component", sig)
rec.check("(g) real S21 map reproduces S21's published dipole",
          abs(g_mult["amp"] - 0.01554) < sig and angle(g_mult["d"], pub) < 2,
          f"{g_mult['amp']:.5f} toward ({g_mult['l']:.1f}, {g_mult['b']:.1f}); published 0.01554 toward (238.2, 28.8)")

rh = fit_dipole_lsq_mult(lam, mask, ae)
rec.check("(h) multiplicative estimator recovers the mock exactly",
          np.linalg.norm(rh["d"] - d_true) < 1e-10 and abs(rh["s"] / s - 1) < 1e-8,
          f"|d| = {rh['amp']/D:.8f} x D; s = {rh['s']:.6e}")

print("  (i) additive bias vs ecliptic latitude of the dipole:")
rows = {}
for el in (-11.15, 0.0, 30.0, 60.0, 89.0):
    c = SkyCoord(171.67 * u.deg, el * u.deg, frame="barycentricmeanecliptic").galactic
    v = hp.ang2vec(c.l.deg, c.b.deg, lonlat=True); ct = v @ pixel_vectors()
    lm = N0 * (1 + s * ae) * (1 + D * ct); lm[mask] = 0
    rows[el] = (fit_dipole_lsq(lm, mask, ecl_t)["amp"] / D, fit_dipole_lsq_mult(lm, mask, ae)["amp"] / D)
    print(f"     ecliptic lat {el:+6.1f}: additive {rows[el][0]:.4f}, multiplicative {rows[el][1]:.4f}")
rec.number("additive_bias_vs_ecliptic_latitude", {str(k): v for k, v in rows.items()})
rec.check("(i) additive bias changes sign toward the ecliptic pole", rows[-11.15][0] > 1 and rows[89.0][0] < 1)
rec.write()
