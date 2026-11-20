"""
Stage s02: the input numbers for the mocks. Prototype Step 2.

Writes results/s02_inputs/inputs.json (read by later stages).

Checks
  * method check: our ecliptic-trend fit applied to the S21 map reproduces
    S21's published slope -0.051 and intercept 68.89 per deg^2 (within 2 sigma)
  * our ecliptic latitudes match those stored in the S21 map
  * S22 flux limit 0.078 mJy
  * x and alpha at the flux limit agree with S22's 1.89 and 1.06
  * D_kin from S22's x and alpha agrees with S22's 0.73e-2
  * the fitted trend averaged over the footprint equals the observed mean
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, healpy as hp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import config
from dipole import ecliptic_latitude_map, load_mask, write_json
from dipole.catalog import load_catalog, galactic_pixels, s22_selection, count_map
from dipole.masks import s21_map
from dipole.measure import ecliptic_trend_fit, alpha_at_limit, x_at_limit, d_kin
from dipole.checks import StageRecord

rec = StageRecord("s02_inputs")
NPIX = hp.nside2npix(config.NSIDE)
PIX_AREA = hp.nside2pixarea(config.NSIDE, degrees=True)
out = {}

beta = config.V_CMB_KMS / config.C_KMS
out.update(v_cmb_kms=config.V_CMB_KMS, l_cmb_deg=config.L_CMB_DEG, b_cmb_deg=config.B_CMB_DEG,
           beta=beta, nside=config.NSIDE, pixel_area_deg2=PIX_AREA)

abselat_pix = np.abs(ecliptic_latitude_map())
mask = load_mask("s22"); seen = ~mask

cat = load_catalog()
sel = s22_selection(cat)
ipix = galactic_pixels(cat["ra"][sel], cat["dec"][sel])
count = count_map(ipix)
in_fp = seen[ipix]
N_tot = int(count[seen].sum()); nbar = count[seen].mean()
out.update(n_quasars_footprint=N_tot, n_pixels_footprint=int(seen.sum()),
           f_sky=float(seen.mean()), mean_count_per_pixel=nbar, mean_density_deg2=nbar / PIX_AREA)

# ---- method check on the S21 map
s21 = s21_map(); m21 = np.asarray(s21["mask"], bool)
p21, e21, bins21 = ecliptic_trend_fit(np.abs(s21["elat"][~m21]), s21["density"][~m21])
rec.check("S21 slope reproduced (published -0.051)", abs(p21[0] + 0.051) < 2 * e21[0],
          f"{p21[0]:.4f} +/- {e21[0]:.4f}")
rec.check("S21 intercept reproduced (published 68.89)", abs(p21[1] - 68.89) < 2 * e21[1],
          f"{p21[1]:.2f} +/- {e21[1]:.2f}")
dmax = float(np.max(np.abs(np.abs(s21["elat"]) - abselat_pix)))
rec.check("ecliptic latitudes match the S21 map", dmax < 1e-8, f"max difference {dmax:.1e} deg")

# ---- S22 trend (counts per pixel, as S22's code)
p22, e22, bins22 = ecliptic_trend_fit(abselat_pix[seen], count[seen])
slope, intercept = p22; frac = slope / intercept
out.update(ecl_slope_count_per_deg=slope, ecl_slope_err=e22[0], ecl_intercept_count=intercept,
           ecl_intercept_err=e22[1], ecl_frac_slope_per_deg=frac, ecl_frac_change_0_to_90=frac * 90,
           s21_check_slope_deg2_per_deg=p21[0], s21_check_intercept_deg2=p21[1])
pred_mean = (intercept * (1 + frac * abselat_pix[seen])).mean()
out["ecl_trend_mean_over_footprint"] = pred_mean
rec.check("fitted trend averages to the observed mean", abs(pred_mean / nbar - 1) < 1e-3,
          f"{pred_mean:.3f} vs {nbar:.3f}")

# ---- x and alpha at the flux limit (S22 method)
a_fp, s_fp = cat["alpha"][sel][in_fp], cat["logS"][sel][in_fp]
logS_min = s_fp.min()
alpha_lim, (sc_, ac_, pa, k) = alpha_at_limit(s_fp, a_fp)
x_lim, (uS, logN, px, k2) = x_at_limit(s_fp)
out.update(logS_min_mJy=logS_min, S_min_mJy=10**logS_min, alpha_ours=alpha_lim, x_ours=x_lim,
           alpha_mean_all=float(a_fp.mean()))
rec.check("flux limit ~ 0.078 mJy (S22)", abs(10**logS_min - 0.078) < 0.001, f"{10**logS_min:.4f} mJy")
rec.check("alpha at limit ~ 1.06 (S22)", abs(alpha_lim - config.ALPHA_S22) < 0.01, f"{alpha_lim:.3f}")
rec.check("x at limit ~ 1.89 (S22)", abs(x_lim - config.X_S22) < 0.02, f"{x_lim:.3f}")

var = {}
for smax in (-0.9, -1.0, -0.8):
    for deg in (2, 3, 4):
        var[f"x_logSmax{smax}_deg{deg}"] = x_at_limit(s_fp, logS_max=smax, deg=deg)[0]
    var[f"alpha_logSmax{smax}"] = alpha_at_limit(s_fp, a_fp, logS_max=smax)[0]
out["fit_choice_variations"] = var
xs = [v for kk, v in var.items() if kk.startswith("x_")]
als = [v for kk, v in var.items() if kk.startswith("alpha_")]

D_pub, D_ours = d_kin(config.X_S22, config.ALPHA_S22, beta), d_kin(x_lim, alpha_lim, beta)
out.update(D_kin_from_published=D_pub, D_kin_ours=D_ours,
           D_kin_ours_range=[d_kin(min(xs), min(als), beta), d_kin(max(xs), max(als), beta)])
rec.check("D_kin from S22's x, alpha = 0.73e-2", abs(D_pub - 0.0073) < 5e-5, f"{D_pub:.5f}")

write_json(out, rec.dir / "inputs.json")
for kk in ("n_quasars_footprint", "mean_count_per_pixel", "ecl_intercept_count", "ecl_frac_slope_per_deg",
           "alpha_ours", "x_ours", "D_kin_from_published", "D_kin_ours"):
    rec.number(kk, out[kk])

# ---- plots
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for a_, (xc, yc, se_), p_, ttl, yl in [
        (ax[0], bins21, p21, "S21 map (check vs published fit)", "density [deg$^{-2}$]"),
        (ax[1], bins22, p22, "S22 sample (our fit)", "count per pixel")]:
    a_.errorbar(xc, yc, se_, fmt=".", ms=3)
    a_.plot(xc, np.polyval(p_, xc), "k--", label=f"slope {p_[0]:.4f}, intercept {p_[1]:.2f}")
    a_.set_xlabel("|ecliptic latitude| [deg]"); a_.set_ylabel(yl); a_.set_title(ttl); a_.grid(alpha=0.4)
ax[0].plot(bins21[0], -0.051 * bins21[0] + 68.89, "r:", label="S21 published")
for a_ in ax: a_.legend()
plt.tight_layout(); plt.savefig(rec.dir / "ecliptic_fit.png", dpi=110); plt.close()
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].plot(sc_, ac_, ".", ms=3); ax[0].plot(sc_[k], np.polyval(pa, sc_[k]), "k--")
ax[0].axvline(logS_min, c="r", ls=":"); ax[0].plot(logS_min, alpha_lim, "r*", ms=12, label=f"alpha at limit {alpha_lim:.3f}")
ax[0].set_xlabel("log S [mJy]"); ax[0].set_ylabel("mean alpha (bins of 10,000)"); ax[0].legend()
ax[1].plot(uS[k2], logN[k2], ".", ms=1); ax[1].plot(uS[k2], np.polyval(px, uS[k2]), "k--")
ax[1].axvline(logS_min, c="r", ls=":"); ax[1].set_xlabel("log S [mJy]"); ax[1].set_ylabel("log N(>S)")
ax[1].set_title(f"x at limit = {x_lim:.3f}")
for a_ in ax: a_.grid(alpha=0.4)
plt.tight_layout(); plt.savefig(rec.dir / "x_alpha.png", dpi=110); plt.close()
rec.write()
