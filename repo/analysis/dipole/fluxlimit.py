"""
x and alpha at the flux limit, and the kinematic dipole they predict
(project Step 3).

Three independent ways to get D_kin from a catalog cut at W1 < m_lim:

1. S22's method (dipole.measure): polynomial fits to log N(>S) and to binned
   alpha over log S < logS_max, evaluated at the faintest flux.
2. Local method (this module): the slope of ln N(<m) and the mean alpha in a
   narrow window of magnitude around the cut. "Two-sided" windows use the
   catalog's sources fainter than the cut (the released catalog goes to
   W1 = 16.6); "one-sided" windows use only the sample itself.
3. Direct boost (this module): Doppler-shift every source's magnitude by its
   own alpha, recount above the cut in the forward and backward directions,
   and include aberration. No x is fitted and the Ellis-Baldwin formula is
   not used (the method of Mittal et al. 2024).

Why x is measured in magnitude. With S proportional to nu^-alpha, motion
changes a source's observed magnitude by -2.5 (1 + alpha) log10(delta) and
the solid angle by delta^-2. Expanding the count above the cut to first order
in beta gives exactly
    D = beta [2 + x (1 + <alpha>_lim)],   x = (d ln N(<m)/dm) / (0.4 ln 10),
with <alpha>_lim the mean alpha of sources AT the cut. (In this catalog
log S = const - 0.4 W1 exactly, so x in log S and in W1 are the same thing.)

Magnitudes are handled as integer millimagnitudes. The catalog stores W1 as
float32 with 0.001 mag resolution; binning the float values directly puts
sources on the wrong side of bin edges and makes histograms lumpy by ~5%.
"""
import numpy as np

LN10_04 = 0.4 * np.log(10.0)


def millimag(w1):
    """W1 magnitudes as integer millimagnitudes."""
    return np.rint(np.asarray(w1, dtype=float) * 1000.0).astype(np.int64)


def _cumulative_below(mm, grid):
    """N(m < g) for every g in `grid` (both integer millimag)."""
    s = np.sort(mm)
    return np.searchsorted(s, grid, side="left")


def x_local(mm, m_lim=16500, half=50, deg=2, one_sided=False):
    """x from a polynomial fit of ln N(<m) against m over the window
    [m_lim - half, m_lim + half] (millimag), or [m_lim - half, m_lim] if
    one_sided; x = slope at m_lim / (0.4 ln 10)."""
    hi = m_lim if one_sided else m_lim + half
    g = np.arange(m_lim - half, hi + 1)
    N = _cumulative_below(mm, g).astype(float)
    t = (g - m_lim) / 1000.0
    p = np.polyfit(t, np.log(N), deg)
    return float(p[-2] / LN10_04)


def alpha_local(mm, alpha, m_lim=16500, half=50, deg=1, one_sided=False):
    """Mean alpha at the cut: least-squares polynomial in m through every
    source in the window, evaluated at m_lim. One-sided windows use only
    sources brighter than the cut (m < m_lim)."""
    hi = m_lim if one_sided else m_lim + half
    k = (mm >= m_lim - half) & (mm < hi)
    t = (mm[k] - m_lim) / 1000.0
    p = np.polyfit(t, alpha[k], deg)
    return float(p[-1])


def d_direct(mm, alpha, beta, k=10.0, m_lim=16.5, seed=0):
    """Forward-backward count asymmetry for speed k*beta, divided by k.

    Each recorded magnitude is spread uniformly over its 0.001-mag rounding
    interval (seeded) so the shifted cut does not fall between discrete
    values. Source counts are taken as the catalog's own (the catalog's
    dipole changes the sky-averaged counts only at second order in beta)."""
    rng = np.random.default_rng(seed)
    m = mm / 1000.0 + rng.uniform(-0.0005, 0.0005, mm.size)
    b = k * beta
    g = 1.0 / np.sqrt(1.0 - b * b)
    n = []
    for sgn in (+1, -1):
        delta = g * (1.0 + sgn * b)
        shift = 2.5 * (1.0 + alpha) * np.log10(delta)
        n.append(delta ** 2 * np.count_nonzero(m < m_lim + shift))
    return float((n[0] - n[1]) / (n[0] + n[1]) / k)


def d_direct_extrapolated(mm, alpha, beta, ks=(5.0, 10.0, 20.0), m_lim=16.5, seed=0):
    """D/k is even in k: fit a + b k^2 to the values at `ks` and return
    a (the k -> 0 limit), the individual values, and the fit residuals."""
    v = np.array([d_direct(mm, alpha, beta, k, m_lim, seed) for k in ks])
    A = np.vstack([np.ones(len(ks)), np.square(ks)]).T
    c, *_ = np.linalg.lstsq(A, v, rcond=None)
    return float(c[0]), v, v - A @ c


# ---------------------------------------------------------------- fake catalogs
class FakeCatalog:
    """Smooth model of the catalog's magnitudes and spectral indices, built
    from the real data, with x and alpha at the cut known exactly.

    * Sources brighter than m_join are copied from the real catalog.
    * Fainter sources (m_join to m_max) are drawn from n(m) = exp(poly(m)),
      a polynomial of degree `deg_n` fitted to the real counts in 0.01-mag bins.
    * alpha = abar(m) + residual, abar a polynomial of degree `deg_alpha`
      fitted to the real sources between m_join and m_max, residuals
      resampled from the real ones. (The real mean alpha curves with
      magnitude, so deg_alpha >= 2 is needed for a realistic test.)
    """

    def __init__(self, mm_real, alpha_real, m_join=15.0, m_max=16.6, m_lim=16.5, deg_n=3, deg_alpha=3):
        self.m_join, self.m_max, self.m_lim = m_join, m_max, m_lim
        self.deg_n, self.deg_alpha = deg_n, deg_alpha
        mj, mx = int(round(m_join * 1000)), int(round(m_max * 1000))
        bright = mm_real < mj
        self.mm_bright, self.alpha_bright = mm_real[bright], alpha_real[bright]
        faint = (mm_real >= mj) & (mm_real < mx)
        cnt = np.bincount((mm_real[faint] - mj) // 10, minlength=(mx - mj) // 10)
        mc = m_join + 0.005 + 0.01 * np.arange(cnt.size)
        p = np.polyfit(mc - m_lim, np.log(cnt / 0.01), deg_n, w=np.sqrt(cnt))
        self.p_logn = p
        self.p_alpha = np.polyfit(mm_real[faint] / 1000.0 - m_lim, alpha_real[faint], deg_alpha)
        self.resid = alpha_real[faint] - np.polyval(self.p_alpha, mm_real[faint] / 1000.0 - m_lim)
        # truth
        self.grid = np.linspace(m_join, m_max, 160001)
        dens = np.exp(np.polyval(p, self.grid - m_lim))
        self.cdf = np.concatenate([[0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(self.grid))])
        n_lim = np.exp(np.polyval(p, 0.0))
        N_lim = bright.sum() + np.interp(m_lim, self.grid, self.cdf)
        self.x_true = float(n_lim / N_lim / LN10_04)
        self.alpha_true = float(self.p_alpha[-1])

    def draw(self, rng):
        """One fake catalog: (millimag, alpha) and continuous magnitudes of
        the faint part (for boosting)."""
        nf = rng.poisson(self.cdf[-1])
        m = np.interp(rng.uniform(0, self.cdf[-1], nf), self.cdf, self.grid)
        a = np.polyval(self.p_alpha, m - self.m_lim) + rng.choice(self.resid, nf)
        mb = self.mm_bright / 1000.0 + rng.uniform(-0.0005, 0.0005, self.mm_bright.size)
        m_all = np.concatenate([mb, m])
        a_all = np.concatenate([self.alpha_bright, a])
        return m_all, a_all
