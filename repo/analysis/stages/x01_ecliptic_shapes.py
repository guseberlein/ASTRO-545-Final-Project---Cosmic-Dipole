"""
Exploration x01 (after the prototype; to be expanded in the shape study):
what if the scanning pattern is NOT a straight line in |ecliptic latitude|?

Part 1 - noiseless mocks: plant several pattern shapes (~8% plane to pole,
plus D_inj), fit with the straight-line template (A) and with none (B).
s08 showed that averages over noisy mocks equal the noiseless fit, so a
noiseless map gives each bias directly.
Part 2 - real S22 data (no NVSS removal): how well |elat| and the WISE
coverage map (1/w1cov) describe the density; the dipole with each template.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, healpy as hp
import config
from dipole import (unit_vector, cos_theta_map, ecliptic_latitude_map, load_mask, load_inputs,
                    fit_dipole_lsq, fit_dipole_lsq_mult, NPIX)
from dipole.catalog import load_catalog, galactic_pixels, s22_selection, count_map
from dipole.checks import StageRecord

rec = StageRecord("x01_ecliptic_shapes")
I = load_inputs()
D = I["D_kin_from_published"]; dh = unit_vector(); cth = cos_theta_map(dh)
elat = ecliptic_latitude_map(); ae = np.abs(elat); mask = load_mask("s22"); seen = ~mask
N0, s = I["ecl_intercept_count"], I["ecl_frac_slope_per_deg"]; contrast = s * 90

cat = load_catalog(("ra", "dec", "w1", "w1cov"))
sel = s22_selection(cat)
ip = galactic_pixels(cat["ra"][sel], cat["dec"][sel])
cnt = count_map(ip).astype(float)
cov = count_map(ip, cat["w1cov"][sel].astype(float)) / np.maximum(cnt, 1)
inv = np.zeros(NPIX); inv[seen] = 1 / cov[seen]
rec.number("w1cov_footprint_median_5_95", [np.median(cov[seen]), np.percentile(cov[seen], 5), np.percentile(cov[seen], 95)])

lin = (1 + s * ae)[seen]; t = inv[seen]; k = lin.std() / t.std()
cv = np.ones(NPIX); cv[seen] = 1 + k * (t - t.mean())
shapes = {
    "straight line in |elat|": 1 + s * ae,
    "cos(elat)": 1 + contrast * (1 - np.cos(np.radians(ae))),
    "quadratic in |elat|": 1 + contrast * (ae / 90) ** 2,
    "step: shallower below 20 deg": 1 + contrast * (ae > 20),
    "straight line + 0.5% N-S difference": (1 + s * ae) * (1 + 0.005 * np.sign(elat)),
    "coverage map ~1/w1cov (size assumed)": cv,
}
ang = lambda r: float(np.degrees(np.arccos(r["d"] @ dh / r["amp"])))
part1 = {}
for name, f in shapes.items():
    lam = N0 * f * (1 + D * cth); lam[mask] = 0
    A = fit_dipole_lsq_mult(lam, mask, ae); B = fit_dipole_lsq(lam, mask)
    part1[name] = dict(A_over_D=A["amp"] / D, A_angle=ang(A), B_over_D=B["amp"] / D, B_angle=ang(B))
    print(f"  {name:40s} A {A['amp']/D:.3f} ({ang(A):4.1f} deg) | B {B['amp']/D:.3f} ({ang(B):4.1f} deg)")
rec.number("noiseless_shapes", part1)

X = np.vstack([np.ones(seen.sum()), ae[seen], inv[seen] * 160]).T
y = cnt[seen] / cnt[seen].mean()
for cols, lab in [([0, 1], "elat"), ([0, 2], "coverage"), ([0, 1, 2], "both")]:
    b = np.linalg.lstsq(X[:, cols], y, rcond=None)[0]
    rec.number(f"density_fit_{lab}", dict(coefficients=b, leftover_variance=float(np.var(y - X[:, cols] @ b))))
real = {}
for lab, f in [("straight line, multiplicative", fit_dipole_lsq_mult(cnt, mask, ae)),
               ("coverage, multiplicative", fit_dipole_lsq_mult(cnt, mask, inv * 160)),
               ("straight line, additive", fit_dipole_lsq(cnt, mask, {"e": ae})),
               ("coverage, additive", fit_dipole_lsq(cnt, mask, {"c": inv})),
               ("both, additive", fit_dipole_lsq(cnt, mask, {"e": ae, "c": inv})),
               ("no template", fit_dipole_lsq(cnt, mask))]:
    real[lab] = dict(D=f["amp"], l=f["l"], b=f["b"], d=f["d"])
    print(f"  real, {lab:32s}: D = {f['amp']:.5f} toward ({f['l']:.1f}, {f['b']:.1f})")
rec.number("real_dipole_by_template", real)
c = inv[seen]; bb = np.polyfit(ae[seen], c, 1); res = c - np.polyval(bb, ae[seen])
rec.number("coverage_variance_explained_by_elat_line", float(1 - np.var(res) / np.var(c)))
a, b = real["straight line, multiplicative"]["d"], real["coverage, multiplicative"]["d"]
rec.number("real_shift_line_to_coverage", dict(angle_deg=float(np.degrees(np.arccos(a @ b / np.linalg.norm(a) / np.linalg.norm(b)))),
                                               vector_length=float(np.linalg.norm(a - b))))
rec.note("Exploratory. The coverage pattern's size in Part 1 is assumed (matched to the straight line's spread).")
rec.write()
