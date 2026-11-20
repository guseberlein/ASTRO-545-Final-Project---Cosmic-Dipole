"""
dipole: tools for measuring the number-count dipole of a HEALPix sky map and
testing the estimator on mock skies.

Conventions used everywhere:
  * HEALPix N_side = 64, RING ordering, Galactic coordinates.
  * Masks are boolean arrays with True = MASKED (thrown away), as in the
    Secrest et al. code; the footprint is ~mask.
  * Dipole vectors are Galactic Cartesian (x, y, z) and dimensionless
    (fractional density modulation).
"""
from .geometry import (pixel_vectors, unit_vector, cos_theta_map,
                       ecliptic_latitude_map, galactic_latitude_map, NPIX)
from .estimators import fit_dipole_lsq, fit_dipole_lsq_mult, fit_multipoles_mult, direction
from .likelihood import amplitude_likelihood
from .mocks import expected_counts, draw_mock
from .io import load_mask, save_mask, load_inputs, write_json, read_json
