"""
Project-wide settings: file locations, physical constants and fixed choices.

By default the data are looked for in the folder that contains this code
folder, so the analysis runs from any working directory with this layout:

    Dipole Project/
        analysis/                      <- this code
        secrest+22_accepted/
            secrest+22_accepted/       <- Secrest et al. (2022) Zenodo release
        quaia/                         <- Quaia catalogs and selection functions

To keep the data elsewhere, set the environment variable DIPOLE_DATA to the
folder that holds secrest+22_accepted/ and quaia/. Results are always written
to results/ inside the code folder.
"""
import os
from pathlib import Path

# ----------------------------------------------------------------- locations
ROOT = Path(__file__).resolve().parent            # .../Dipole Project/analysis
PROJECT = Path(os.environ.get("DIPOLE_DATA", ROOT.parent)).resolve()   # .../Dipole Project
RELEASE = PROJECT / "secrest+22_accepted" / "secrest+22_accepted"
CATALOG = RELEASE / "wise" / "reference" / "catwise_agns.fits"
EXCLUDE_REGIONS = RELEASE / "wise" / "reference" / "exclude_master_revised.fits"
S21_MAP = RELEASE / "secrest+21" / "secrest+21" / "secrest+21_nside64.fits"
RESULTS = ROOT / "results"
PROTOTYPE = PROJECT / "prototype"                 # week-11 prototype (kept as a record)

# ----------------------------------------------------------------- sky grid
NSIDE = 64                     # HEALPix resolution used by S21 and S22

# ----------------------------------------------------------------- CMB dipole (Planck 2018, as used by S21/S22)
C_KMS = 299792.458
V_CMB_KMS = 369.82
L_CMB_DEG, B_CMB_DEG = 264.021, 48.253

# ----------------------------------------------------------------- S22 sample definition (wise/code/1_mk_mask.py)
W1_CUT = 16.5                  # keep W1 < 16.5 (Vega mag)
W1COV_MIN = 80                 # keep w1cov >= 80
GAL_CUT_DEG = 30.0             # mask |b| < 30 deg

# ----------------------------------------------------------------- published S22 values (arXiv:2206.05624)
X_S22, ALPHA_S22 = 1.89, 1.06


def stage_dir(name):
    """Output folder for one stage, created if needed."""
    d = RESULTS / name
    d.mkdir(parents=True, exist_ok=True)
    return d
