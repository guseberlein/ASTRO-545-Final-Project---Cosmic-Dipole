"""Sky geometry on the HEALPix grid (prototype Step 3)."""
import numpy as np
import healpy as hp
from astropy.coordinates import SkyCoord
import astropy.units as u

import config

NSIDE = config.NSIDE
NPIX = hp.nside2npix(NSIDE)


def pixel_vectors(nside=NSIDE):
    """Unit vector of every pixel centre, shape (3, npix), Galactic Cartesian."""
    return np.array(hp.pix2vec(nside, np.arange(hp.nside2npix(nside))))


def unit_vector(l_deg=config.L_CMB_DEG, b_deg=config.B_CMB_DEG):
    """Unit vector pointing to Galactic (l, b); default = CMB dipole direction."""
    return hp.ang2vec(l_deg, b_deg, lonlat=True)


def cos_theta_map(direction_vec, nside=NSIDE):
    """cos of the angle between each pixel centre and `direction_vec`."""
    return direction_vec @ pixel_vectors(nside)


def galactic_latitude_map(nside=NSIDE):
    """Galactic latitude b (deg) of every pixel centre."""
    return hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)[1]


def ecliptic_latitude_map(nside=NSIDE):
    """Ecliptic latitude (deg, barycentric mean ecliptic) of every pixel centre."""
    l, b = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)
    sc = SkyCoord(l * u.deg, b * u.deg, frame="galactic")
    return sc.barycentricmeanecliptic.lat.deg
