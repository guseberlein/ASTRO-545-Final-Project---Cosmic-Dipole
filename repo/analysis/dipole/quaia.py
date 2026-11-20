"""
Quaia (project Step 7).

Quaia's selection function S is a HEALPix map in EQUATORIAL coordinates, so
the Quaia analysis works on the equatorial grid (as Mittal et al. 2024 did):
counts are binned by (ra, dec), Galactic cuts use each pixel centre's
Galactic latitude, and fitted dipole vectors are converted to Galactic
coordinates only for reporting.

Model for the counts in pixel p (S fixed, as in Mittal et al.):
    lam_p = a S_p (1 + sum_k s_k t_k(p)) (1 + d . n_p + q . Y(p))
fit_offset_lsq   weighted least squares (weights 1/S, i.e. Poisson variance),
                 iterated for the template amplitudes s; fast, for mocks
fit_offset_ml    Poisson maximum likelihood of the same model (Mittal et al.'s
                 likelihood); for the real data

Spectral index (S_nu ~ nu^-alpha) from two Gaia bands, using the AB - Vega
offsets and pivot wavelengths of Riello et al. (2021, Table 3).
"""
import numpy as np
import healpy as hp

from .geometry import NPIX, NSIDE

AB_MINUS_VEGA = {"G": 25.8010 - 25.6874, "BP": 25.3540 - 25.3385, "RP": 25.1040 - 24.7479}
PIVOT_NM = {"G": 621.79, "BP": 510.97, "RP": 776.91}

_ROT_C2G = hp.Rotator(coord=["C", "G"])
_ROT_G2C = hp.Rotator(coord=["G", "C"])


def eq_pixel_vectors():
    return np.array(hp.pix2vec(NSIDE, np.arange(NPIX)))


def galactic_latitude_of_eq_pixels():
    th, ph = _ROT_C2G(*hp.pix2ang(NSIDE, np.arange(NPIX)))
    return 90.0 - np.degrees(th)


def galactic_longitude_of_eq_pixels():
    th, ph = _ROT_C2G(*hp.pix2ang(NSIDE, np.arange(NPIX)))
    return np.degrees(ph) % 360


def eq_vector(l_deg, b_deg):
    """Equatorial unit vector of a Galactic direction."""
    return _ROT_G2C(hp.ang2vec(l_deg, b_deg, lonlat=True))


def to_galactic(d_eq):
    v = _ROT_C2G(d_eq)
    l, b = hp.vec2ang(v / np.linalg.norm(v), lonlat=True)
    return float(l[0]), float(b[0])


def spectral_index(m1, m2, b1, b2):
    """alpha (S_nu ~ nu^-alpha) between bands b1, b2 from Vega magnitudes."""
    a1, a2 = m1 + AB_MINUS_VEGA[b1], m2 + AB_MINUS_VEGA[b2]
    # S1/S2 = (nu1/nu2)^-alpha = (lambda2/lambda1)^-alpha = 10^(-0.4 (m1 - m2))
    return 0.4 * (a1 - a2) / np.log10(PIVOT_NM[b2] / PIVOT_NM[b1])


def fit_offset_lsq(counts, mask, S, extra=None, trends=None, n_iter=50, tol=1e-12):
    seen = ~mask
    n = counts[seen].astype(float); s_ = S[seen]
    V = eq_pixel_vectors()[:, seen]
    Y = np.zeros((0, seen.sum())) if extra is None else np.asarray(extra, float)[:, seen]
    T = np.zeros((0, seen.sum())) if not trends else np.array([np.asarray(t, float)[seen] for t in trends])
    K, nq = T.shape[0], Y.shape[0]
    sc = T.std(axis=1) if K else np.zeros(0)
    s = np.zeros(K)
    w = 1.0 / s_
    for _ in range(n_iter if K else 1):
        mod = s_ * (1 + s @ T) if K else s_
        A = np.vstack([V * mod, Y * mod, s_ * T, s_])
        F = (A * w) @ A.T
        x = np.linalg.solve(F, (A * w) @ n)
        a = x[-1]
        if not K:
            break
        s_new = x[3 + nq:3 + nq + K] / a
        done = np.all(np.abs((s_new - s) * sc) < tol)
        s = s_new
        if done:
            break
    cov = np.linalg.inv(F)
    d = x[:3] / a
    # weights 1/S omit the constant a of the Poisson variance a S, so the covariance is a x inv(F)
    return dict(d=d, amp=float(np.linalg.norm(d)), a=a, s=s, q=x[3:3 + nq] / a,
                sigma=np.sqrt(a * np.diag(cov[:3, :3])) / a)


def fit_offset_ml(counts, mask, S, extra=None, trends=None):
    """Poisson maximum likelihood, started from the least-squares solution."""
    from scipy.optimize import minimize
    seen = ~mask
    n = counts[seen].astype(float); s_ = S[seen]
    V = eq_pixel_vectors()[:, seen].T
    Y = np.zeros((seen.sum(), 0)) if extra is None else np.asarray(extra, float)[:, seen].T
    T = np.zeros((seen.sum(), 0)) if not trends else np.array([np.asarray(t, float)[seen] for t in trends]).T
    K, nq = T.shape[1], Y.shape[1]
    f0 = fit_offset_lsq(counts, mask, S, extra, trends)
    x0 = np.r_[np.log(f0["a"]), f0["s"], f0["d"], f0["q"]]

    def nll(x):
        A = np.exp(x[0]); s = x[1:1 + K]; d = x[1 + K:4 + K]; q = x[4 + K:]
        m1 = 1 + T @ s; m2 = 1 + V @ d + Y @ q
        lam = A * s_ * m1 * m2
        r = n / lam - 1
        g = np.r_[-np.sum(n - lam),
                  -(T.T @ (r * A * s_ * m2)),
                  -(V.T @ (r * A * s_ * m1)),
                  -(Y.T @ (r * A * s_ * m1))]
        return -np.sum(n * np.log(lam) - lam), g
    res = minimize(nll, x0, jac=True, method="L-BFGS-B", options=dict(maxiter=5000, ftol=1e-15, gtol=1e-10))
    d = res.x[1 + K:4 + K]
    return dict(d=d, amp=float(np.linalg.norm(d)), s=res.x[1:1 + K], a=float(np.exp(res.x[0])),
                converged=bool(res.success), lsq=f0)
