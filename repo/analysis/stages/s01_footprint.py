"""
Stage s01: the footprint (mask). Prototype Step 1.

Builds three masks (True = masked): the |b| < 30 fallback, the S21 mask
stored in the S21 sky map, and the S22 mask rebuilt from S22's recipe.

Checks
  * fallback sky fraction = 1 - sin 30 deg = 0.500 (tests the latitude cut)
  * the S21 map uses our pixel grid (latitudes identical)
  * S21 map counts sum to S21's published 1,355,352 quasars
  * rebuilt S22 mask: ~51% of sky masked (S22: "51% of the sky was masked")
  * S22 sample in footprint ~1.6 million (S22: "1.6 million", before its
    ~1.4% NVSS-counterpart removal)
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, healpy as hp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import config
from dipole import galactic_latitude_map, save_mask
from dipole.masks import fallback_mask, s21_map, rebuild_s22_mask
from dipole.catalog import load_catalog, galactic_pixels, count_map
from dipole.checks import StageRecord

rec = StageRecord("s01_footprint")
NPIX = hp.nside2npix(config.NSIDE)

m_fb = fallback_mask()
rec.number("fallback_fsky", float(np.mean(~m_fb)))
rec.check("fallback sky fraction = 0.500", abs(np.mean(~m_fb) - 0.5) < 1e-3, f"{np.mean(~m_fb):.4f}")

s21 = s21_map()
rec.check("S21 map rows are in pixel order", np.all(s21["ipix"] == np.arange(NPIX)))
rec.check("S21 map uses our Galactic RING grid", np.allclose(s21["b"], galactic_latitude_map(), atol=1e-6))
m21 = np.asarray(s21["mask"], dtype=bool)
n21_in, n21_out = int(s21["count"][~m21].sum()), int(s21["count"][m21].sum())
rec.number("s21_fsky", float(np.mean(~m21)))
rec.number("s21_quasars_in_footprint", n21_in)
rec.number("s21_quasars_in_masked_pixels", n21_out)
rec.check("S21 map total = published 1,355,352", n21_in + n21_out == 1355352,
          f"{n21_in:,} in footprint + {n21_out:,} in masked pixels = {n21_in + n21_out:,}")

cat = load_catalog(("ra", "dec", "w1", "w1cov"))
rec.number("catalog_rows", int(cat["ra"].size))
good = cat["w1cov"] >= config.W1COV_MIN
ip_good = galactic_pixels(cat["ra"][good], cat["dec"][good])
m22, info = rebuild_s22_mask(ip_good)
for k, v in info.items():
    rec.number("s22_mask_" + k, v)
sel = good & (cat["w1"] < config.W1_CUT)
cnt = count_map(galactic_pixels(cat["ra"][sel], cat["dec"][sel]))
n22 = int(cnt[~m22].sum())
rec.number("s22_fsky", float(np.mean(~m22)))
rec.number("s22_pixels_kept", int((~m22).sum()))
rec.number("s22_quasars_in_footprint_before_nvss_removal", n22)
rec.check("S22 masked fraction ~ 51%", 0.50 <= np.mean(m22) <= 0.52, f"{100*np.mean(m22):.1f}% masked")
rec.check("S22 sample ~ 1.6 million", 1.55e6 <= n22 <= 1.65e6, f"{n22:,} quasars with W1 < {config.W1_CUT}")

both = (~m21) & (~m22)
rec.number("pixels_kept_by_both", int(both.sum()))
rec.number("pixels_kept_only_by_s21", int(((~m21) & m22).sum()))
rec.number("pixels_kept_only_by_s22", int((m21 & ~m22).sum()))
rec.note(f"S22 applies {info['regions_used']} of the {info['regions_listed']} listed regions; "
         "S21 masked 291 regions (S21 sec. 2), so S22 keeps many small areas S21 removed.")

for name, m in [("fallback", m_fb), ("s21", m21), ("s22", m22)]:
    save_mask(m, name)

fig = plt.figure(figsize=(14, 4))
for i, (name, m) in enumerate([("Fallback |b|<30", m_fb), ("S21 mask (from release)", m21),
                               ("S22 mask (rebuilt)", m22)]):
    hp.mollview((~m).astype(float), fig=fig.number, sub=(1, 3, i + 1), cmap="Greys_r", cbar=False,
                min=0, max=1, title=f"{name}\nkept fraction {np.mean(~m):.3f}")
plt.savefig(rec.dir / "masks.png", dpi=110, bbox_inches="tight"); plt.close()
diff = np.zeros(NPIX); diff[(~m21) & m22] = 1; diff[m21 & ~m22] = -1
hp.mollview(diff, title="+1 kept only by S21, -1 kept only by S22", cmap="coolwarm", min=-1, max=1)
plt.savefig(rec.dir / "mask_difference.png", dpi=110, bbox_inches="tight"); plt.close()
rec.write()
