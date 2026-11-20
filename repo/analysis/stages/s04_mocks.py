"""
Stage s04: mock generator checks. Prototype Step 4.

Checks (1,000 mocks)
  * the noiseless expected map has the real sample's mean level and contains
    exactly the planted dipole and ecliptic factors; masked pixels are 0
  * each pixel's average over mocks equals its expected count
  * each pixel's variance equals its mean (Poisson)
  * the total count per mock has mean sum(lambda) and spread sqrt(sum(lambda))
    (with 1,000 mocks the spread came out 2.2 sigma high in the prototype; it
    is therefore checked with 2 x 5,000 mocks, which settled it as chance)
  * masked pixels are always empty
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, healpy as hp
import config
from dipole import (unit_vector, cos_theta_map, ecliptic_latitude_map, load_mask, load_inputs,
                    expected_counts, draw_mock)
from dipole.checks import StageRecord

rec = StageRecord("s04_mocks")
I = load_inputs()
mask = load_mask("s22"); seen = ~mask
cth = cos_theta_map(unit_vector()); ae = np.abs(ecliptic_latitude_map())
kw = dict(N0=I["ecl_intercept_count"], s=I["ecl_frac_slope_per_deg"], D=I["D_kin_from_published"])
lam = expected_counts(cth, ae, mask, **kw)
lam_ecl = expected_counts(cth, ae, mask, dipole=False, **kw)
lam_dip = expected_counts(cth, ae, mask, ecliptic=False, **kw)
D, s = kw["D"], kw["s"]

rec.check("expected map has the real sample's mean level", abs(lam[seen].mean() / I["mean_count_per_pixel"] - 1) < 1e-3,
          f"{lam[seen].mean():.3f} vs {I['mean_count_per_pixel']:.3f}")
e1 = np.max(np.abs(lam[seen] / lam_ecl[seen] - (1 + D * cth[seen])))
e2 = np.max(np.abs(lam[seen] / lam_dip[seen] - (1 + s * ae[seen])))
rec.check("dipole factor is exactly 1 + D cos(theta)", e1 < 1e-12, f"max deviation {e1:.1e}")
rec.check("ecliptic factor is exactly 1 + s|elat|", e2 < 1e-12, f"max deviation {e2:.1e}")
rec.check("masked pixels have zero expectation", np.all(lam[mask] == 0))

NMOCK = 1000
rng = np.random.default_rng(20261031)
s1 = np.zeros(lam.size); s2 = np.zeros_like(s1); masked_nonzero = 0
for i in range(NMOCK):
    m = draw_mock(lam, rng)
    s1 += m; s2 += m.astype(float) ** 2
    masked_nonzero += int((m[mask] != 0).sum())
mean_pix = s1 / NMOCK
var_pix = (s2 - NMOCK * mean_pix**2) / (NMOCK - 1)
L = lam[seen]
z = (mean_pix[seen] - L) / np.sqrt(L / NMOCK)
ratio = var_pix[seen] / L
rec.number("pixel_z_mean", z.mean()); rec.number("pixel_z_std", z.std())
rec.check("pixel averages match expectation (z mean 0, std 1)",
          abs(z.mean()) < 3 / np.sqrt(z.size) and abs(z.std() - 1) < 0.02, f"mean {z.mean():+.4f}, std {z.std():.4f}")
rec.number("var_over_mean_mean", ratio.mean()); rec.number("var_over_mean_spread", ratio.std())
rec.check("pixel variance = mean (Poisson)", abs(ratio.mean() - 1) < 0.002 and abs(ratio.std() / np.sqrt(2 / (NMOCK - 1)) - 1) < 0.05,
          f"ratio {ratio.mean():.4f}, spread {ratio.std():.4f} (expect {np.sqrt(2/(NMOCK-1)):.4f})")
rec.check("masked pixels always empty", masked_nonzero == 0, f"{masked_nonzero} nonzero")

for seed in (1, 2):
    r2 = np.random.default_rng(seed); N = 5000
    t = np.array([draw_mock(lam, r2)[seen].sum() for _ in range(N)])
    sd, err = t.std(ddof=1), np.sqrt(L.sum()) / np.sqrt(2 * (N - 1))
    zs, zm = (sd - np.sqrt(L.sum())) / err, (t.mean() - L.sum()) / np.sqrt(L.sum() / N)
    rec.number(f"total_count_spread_seed{seed}", sd)
    rec.check(f"total count: mean and spread (5,000 mocks, seed {seed})", abs(zs) < 3 and abs(zm) < 3,
              f"spread {sd:.0f} vs {np.sqrt(L.sum()):.0f} ({zs:+.1f} sigma); mean offset {zm:+.1f} sigma")
rec.write()
