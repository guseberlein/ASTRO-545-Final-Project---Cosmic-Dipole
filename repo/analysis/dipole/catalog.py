"""Reading the CatWISE quasar catalog shipped in the S22 release."""
import numpy as np
import healpy as hp
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

import config
from .geometry import NSIDE, NPIX


def load_catalog(columns=("ra", "dec", "w1", "w1cov", "alpha", "logS")):
    """Selected columns of wise/reference/catwise_agns.fits as numpy arrays.
    (The released catalog already has W1 - W2 >= 0.8 and 9 <= W1 < 16.6.)"""
    with fits.open(config.CATALOG, memmap=True) as h:
        d = h[1].data
        return {c: np.array(d[c]) for c in columns}


def galactic_pixels(ra, dec, nside=NSIDE):
    """HEALPix pixel (Galactic, RING) of each ICRS position."""
    g = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs").galactic
    return hp.ang2pix(nside, g.l.deg, g.b.deg, lonlat=True)


def s22_selection(cat, w1_cut=config.W1_CUT, w1cov_min=config.W1COV_MIN):
    """Boolean selection of the S22 sample (before masking and before S22's
    removal of NVSS counterparts)."""
    return (cat["w1cov"] >= w1cov_min) & (cat["w1"] < w1_cut)


def count_map(ipix, weights=None):
    """Counts (or summed weights) per pixel."""
    return np.bincount(ipix, weights=weights, minlength=NPIX)
