"""
Stage s03: geometry checks. Prototype Step 3.

Checks
  * pixel vectors have unit length
  * full-sky average of cos(theta) = 0 (equal-area pixels)
  * cos(theta) ~ +1 / -1 at the CMB-direction pixel / its opposite
  * ecliptic latitude ~ +/-90 deg at the ecliptic poles
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, healpy as hp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import config
from dipole import pixel_vectors, unit_vector, cos_theta_map, ecliptic_latitude_map
from dipole.checks import StageRecord

rec = StageRecord("s03_geometry")
vec = pixel_vectors(); d_cmb = unit_vector(); cth = cos_theta_map(d_cmb); elat = ecliptic_latitude_map()
e = float(np.max(np.abs(np.linalg.norm(vec, axis=0) - 1)))
rec.check("pixel vectors have unit length", e < 1e-12, f"max deviation {e:.1e}")
rec.check("full-sky mean of cos(theta) = 0", abs(cth.mean()) < 1e-12, f"{cth.mean():.1e}")
ip = hp.ang2pix(config.NSIDE, config.L_CMB_DEG, config.B_CMB_DEG, lonlat=True)
ia = hp.vec2pix(config.NSIDE, *(-d_cmb))
rec.check("cos(theta) ~ +1 at CMB pixel, ~ -1 opposite", cth[ip] > 0.9999 and cth[ia] < -0.9999,
          f"{cth[ip]:.6f}, {cth[ia]:.6f}")
inep = hp.ang2pix(config.NSIDE, 96.38, 29.81, lonlat=True)
isep = hp.ang2pix(config.NSIDE, 276.38, -29.81, lonlat=True)
rec.check("ecliptic latitude ~ +/-90 at the ecliptic poles", elat[inep] > 89 and elat[isep] < -89,
          f"{elat[inep]:+.2f}, {elat[isep]:+.2f} deg")
fig = plt.figure(figsize=(12, 3.6))
hp.mollview(cth, fig=fig.number, sub=(1, 2, 1), title="cos(theta) from CMB dipole direction")
hp.mollview(elat, fig=fig.number, sub=(1, 2, 2), title="Ecliptic latitude (deg)", cmap="RdBu_r")
plt.savefig(rec.dir / "geometry.png", dpi=110, bbox_inches="tight"); plt.close()
rec.write()
