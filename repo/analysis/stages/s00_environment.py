"""
Stage s00: confirm the software environment and the input data are in place.

Checks
  * required packages import; versions recorded
  * N_side = 64 gives exactly 12 * 64^2 = 49,152 pixels
  * the S22 release files this analysis reads exist
"""
import sys, pathlib, platform
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, healpy as hp, scipy, astropy, matplotlib
import config
from dipole.checks import StageRecord

rec = StageRecord("s00_environment")
for name, mod in [("python", None), ("numpy", np), ("healpy", hp), ("scipy", scipy),
                  ("astropy", astropy), ("matplotlib", matplotlib)]:
    v = platform.python_version() if mod is None else mod.__version__
    rec.number(f"version_{name}", v)
    print(f"  {name:10s} {v}")

npix = hp.nside2npix(config.NSIDE)
rec.number("npix", npix)
rec.number("pixel_area_deg2", hp.nside2pixarea(config.NSIDE, degrees=True))
rec.check("N_side = 64 gives 49,152 pixels", npix == 12 * config.NSIDE**2 == 49152, f"{npix} pixels")
for label, path in [("CatWISE catalog", config.CATALOG), ("S22 exclusion regions", config.EXCLUDE_REGIONS),
                    ("S21 sky map", config.S21_MAP)]:
    rec.check(f"{label} present", path.exists(), str(path.relative_to(config.PROJECT)))
rec.write()
