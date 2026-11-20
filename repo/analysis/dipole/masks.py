"""
Footprints (prototype Step 1). Convention: True = masked.

The S22 release ships the recipe for its mask (wise/code/1_mk_mask.py), not
a mask file. rebuild_s22_mask re-implements it without S22's extra software:
  1. keep sources with w1cov >= 80; mask every pixel with no sources, and its
     8 neighbours;
  2. mask every pixel touching the circular regions flagged use == True in
     wise/reference/exclude_master_revised.fits (48 of the 292 listed);
  3. mask every pixel whose centre has |b| < 30 deg.
"""
import numpy as np
import healpy as hp
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

import config
from .geometry import NSIDE, NPIX, galactic_latitude_map


def fallback_mask(gal_cut=config.GAL_CUT_DEG):
    """Galactic-plane cut only."""
    return np.abs(galactic_latitude_map()) < gal_cut


def s21_map():
    """The S21 sky map table shipped in the S22 release (real quasar counts)."""
    return fits.open(config.S21_MAP)[1].data


def s21_mask():
    return np.asarray(s21_map()["mask"], dtype=bool)


def rebuild_s22_mask(ipix_w1cov_ok, gal_cut=config.GAL_CUT_DEG, verbose=True):
    """`ipix_w1cov_ok`: pixel of every catalog source with w1cov >= 80
    (no magnitude cut, as in S22's mask_zeros step)."""
    mask = np.zeros(NPIX, dtype=bool)
    counts = np.bincount(ipix_w1cov_ok, minlength=NPIX)
    empty = np.where(counts == 0)[0]
    mask[empty] = True
    if empty.size:
        nb = hp.get_all_neighbours(NSIDE, empty)
        mask[nb[nb >= 0]] = True
    n1 = int(mask.sum())

    ex = fits.open(config.EXCLUDE_REGIONS)[1].data
    n_listed = len(ex)
    ex = ex[ex["use"] == True]
    exg = SkyCoord(ex["ra"] * u.deg, ex["dec"] * u.deg, frame="icrs").galactic
    vecs = hp.ang2vec(exg.l.deg, exg.b.deg, lonlat=True)
    for v, r in zip(vecs, np.radians(ex["radius"])):
        mask[hp.query_disc(NSIDE, v, r, inclusive=True)] = True
    n2 = int(mask.sum())

    mask |= np.abs(galactic_latitude_map()) < gal_cut
    info = dict(empty_pixels=int(empty.size), masked_after_step1=n1,
                regions_listed=n_listed, regions_used=int(len(ex)),
                masked_after_step2=n2, masked_after_step3=int(mask.sum()))
    if verbose:
        print("  S22 mask rebuild:", info)
    return mask, info
