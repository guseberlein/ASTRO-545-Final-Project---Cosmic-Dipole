"""
Clustering for mock skies (project Step 4).

* The quasar redshift distribution, read by eye from S21 Fig. 3 and fitted
  with Tiwari et al.'s form N(z) ~ z^a1 exp[-(z/a2)^a3].
* Theory power spectra from CAMB (Planck 2018, Tiwari et al.'s best-fit bias
  b(z) = 1.54 + 0.53 z + 0.50 z^2), plus an independent exact linear
  calculation for low multipoles (including l = 1, which CAMB does not give).
* A lognormal sky generator: a Gaussian field G with spectrum chosen so that
  delta = exp(G - var/2) - 1 has the target spectrum.
* The spectrum measurement applied identically to data and mocks: fit
  monopole, dipole and the multiplicative ecliptic trend, divide them out,
  and take the pseudo-spectrum of the residual on the footprint.

All spectra are of the fractional overdensity delta, per steradian.
"""
import numpy as np
import healpy as hp

from .geometry import pixel_vectors, NPIX, NSIDE

# Planck 2018 TT,TE,EE+lowE+lensing (as Tiwari et al. 2023, Sec. 2)
PLANCK18 = dict(ombh2=0.02237, omch2=0.1200, cosmomc_theta=1.04092e-2, tau=0.0544,
                As=np.exp(3.044) * 1e-10, ns=0.9649)

# S21 Fig. 3, read by eye (z, PDF). Approximate: used only for the shape of the
# theory spectrum; its amplitude is fitted to the data.
NZ_EYE = np.array([
    [0.05, 0.08], [0.15, 0.15], [0.25, 0.38], [0.35, 0.36], [0.45, 0.42], [0.55, 0.48],
    [0.65, 0.55], [0.75, 0.66], [0.85, 0.74], [0.95, 0.84], [1.05, 0.80], [1.15, 0.74],
    [1.25, 0.68], [1.35, 0.60], [1.45, 0.53], [1.55, 0.47], [1.65, 0.41], [1.75, 0.35],
    [1.85, 0.30], [1.95, 0.25], [2.1, 0.19], [2.3, 0.14], [2.5, 0.10], [2.7, 0.07],
    [2.9, 0.05], [3.1, 0.035], [3.3, 0.02], [3.5, 0.01]])


def nz_model(z, a1, a2, a3):
    z = np.asarray(z, float)
    return np.where(z > 0, z ** a1 * np.exp(-(np.clip(z, 0, None) / a2) ** a3), 0.0)


_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def nz_s21(kind="table"):
    """Normalized N(z) on z = 0..4. kind="table": monotone (PCHIP)
    interpolation of the by-eye S21 points (default; follows the figure,
    including its low-z shoulder). kind="fit": Tiwari's parametric form fitted
    to the same points (smoother). Returns (zg, n, zmean, info)."""
    from scipy.interpolate import PchipInterpolator
    from scipy.optimize import curve_fit
    z, p = NZ_EYE.T
    zg = np.linspace(0, 4, 801)
    if kind == "table":
        zz = np.concatenate([[0.0], z, [3.7]]); pp = np.concatenate([[0.0], p, [0.0]])
        n = np.clip(PchipInterpolator(zz, pp, extrapolate=False)(zg), 0, None)
        n = np.nan_to_num(n)
        info = {}
    else:
        par, _ = curve_fit(nz_model, z, p, p0=(2.0, 0.83, 1.5), maxfev=20000)
        n = nz_model(zg, *par)
        info = dict(a1=par[0], a2=par[1], a3=par[2])
    n /= _trapz(n, zg)
    return zg, n, float(_trapz(zg * n, zg)), info


def bias_tiwari(z):
    """Tiwari et al. (2023) best fit, b(z) = 1.54 + 0.53 z + 0.50 z^2."""
    return 1.54 + 0.53 * z + 0.50 * z ** 2


def camb_cls(zg, nz, bz, lmax=400, dlog10Ndm=0.0, all_terms=True, nonlinear=True):
    """CAMB number-count spectrum C_l (l = 0..lmax; l < 2 set to 0).
    all_terms=False keeps only the density term (for the cross-check)."""
    import camb
    from camb.sources import SplinedSourceWindow
    pars = camb.set_params(**PLANCK18, lmax=lmax)
    pars.Want_CMB = False
    pars.NonLinear = camb.model.NonLinear_both if nonlinear else camb.model.NonLinear_none
    st = pars.SourceTerms
    st.counts_density = True
    for k in ("counts_redshift", "counts_lensing", "counts_velocity", "counts_radial",
              "counts_timedelay", "counts_ISW", "counts_potential"):
        setattr(st, k, bool(all_terms) and k in ("counts_redshift", "counts_lensing"))
    st.counts_evolve = False
    pars.SourceWindows = [SplinedSourceWindow(z=zg[1:], W=nz[1:], bias_z=bz[1:], dlog10Ndm=dlog10Ndm)]
    res = camb.get_results(pars)
    cl = res.get_source_cls_dict(raw_cl=True)["W1xW1"][: lmax + 1].copy()
    cl[:2] = 0.0
    return cl, res


def exact_linear_cls(res, zg, nz, bz, ells=(1, 2, 3, 4, 5, 10), kmin=1e-5, kmax=1.0, nk=3000):
    """Exact (no Limber) linear density-only C_l:
        C_l = (2/pi) int dk k^2 P(k, 0) [int dz N(z) b(z) D(z) j_l(k chi(z))]^2,
    with P(k, 0), D(z) and chi(z) from the same CAMB run. Independent of
    CAMB's source-window code."""
    from scipy.special import spherical_jn
    import camb
    k = np.logspace(np.log10(kmin), np.log10(kmax), nk)                 # 1/Mpc
    PK = camb.get_matter_power_interpolator(camb.set_params(**PLANCK18), zmax=4.1, kmax=kmax * 1.2,
                                            nonlinear=False, hubble_units=False, k_hunit=False)
    p0 = np.ravel(PK.P(0.0, k))
    zz = zg[1:]
    chi = res.comoving_radial_distance(zz)
    D = np.sqrt(np.ravel(PK.P(zz, 1e-3)) / float(np.ravel(PK.P(0.0, 1e-3))[0]))   # linear growth at k = 1e-3/Mpc
    w = nz[1:] * bz[1:] * D
    dz = np.gradient(zz)
    out = {}
    for ell in ells:
        J = spherical_jn(ell, np.outer(k, chi))                          # (nk, nz)
        I = J @ (w * dz)
        out[ell] = float(2 / np.pi * _trapz(k ** 2 * p0 * I ** 2, k))
    return out


# ---------------------------------------------------------------- lognormal skies
def gaussian_cls_for_lognormal(cl, lmax, nquad=2000):
    """C_l of the Gaussian field G such that exp(G - var/2) - 1 has spectrum cl
    (l = 0..lmax). Uses xi_G(theta) = ln(1 + xi(theta))."""
    x, wq = np.polynomial.legendre.leggauss(nquad)
    ell = np.arange(lmax + 1)
    P = np.zeros((lmax + 1, x.size))
    P[0] = 1.0
    if lmax >= 1:
        P[1] = x
    for l in range(2, lmax + 1):
        P[l] = ((2 * l - 1) * x * P[l - 1] - (l - 1) * P[l - 2]) / l
    xi = ((2 * ell + 1) / (4 * np.pi) * cl[: lmax + 1]) @ P
    if np.any(xi <= -1):
        raise ValueError("correlation function <= -1: not a valid lognormal field")
    xig = np.log1p(xi)
    clg = 2 * np.pi * (P * wq) @ xig
    clg[0] = 0.0
    return np.clip(clg, 0, None)


def random_alm(cl, lmax, rng):
    """Gaussian alm with spectrum cl, drawn from a numpy Generator
    (healpy.synfast uses the global numpy RNG, which is not used here)."""
    n = hp.Alm.getsize(lmax)
    l, m = hp.Alm.getlm(lmax)
    c = cl[l]
    re = rng.standard_normal(n); im = rng.standard_normal(n)
    a = np.where(m == 0, np.sqrt(c) * re, np.sqrt(c / 2) * (re + 1j * im))
    return a.astype(complex)


def lognormal_delta(clg, lmax, rng, nside=NSIDE):
    """One lognormal overdensity map (mean 0, delta > -1)."""
    g = hp.alm2map(random_alm(clg, lmax, rng), nside, lmax=lmax, pixwin=False)
    var = np.sum((2 * np.arange(lmax + 1) + 1) / (4 * np.pi) * clg[: lmax + 1])
    return np.expm1(g - var / 2)


# ---------------------------------------------------------------- measuring spectra
BANDS = [(2, 2), (3, 3), (4, 4), (5, 5), (6, 7), (8, 10), (11, 15), (16, 25), (26, 40),
         (41, 70), (71, 120), (121, 191)]


def residual_map(counts, mask, trend):
    """delta = n / model - 1 on the footprint, model = monopole x ecliptic
    trend x dipole fitted with the multiplicative estimator; 0 elsewhere."""
    from .estimators import fit_dipole_lsq_mult
    f = fit_dipole_lsq_mult(counts, mask, trend)
    model = f["a"] * f["nbar"] * (1 + f["s"] * np.asarray(trend, float)) * (1 + pixel_vectors().T @ f["d"])
    seen = ~mask
    out = np.zeros(NPIX)
    out[seen] = counts[seen] / model[seen] - 1
    return out


def pseudo_cl(counts, mask, trend, lmax=191):
    """Pseudo-spectrum of the residual map, divided by the sky fraction."""
    return hp.anafast(residual_map(counts, mask, trend), lmax=lmax) / (~mask).mean()


def band(cl, bands=BANDS):
    return np.array([cl[a:b + 1].mean() for a, b in bands])


def ylm_templates(ells=(2, 3), nside=NSIDE):
    """Real spherical-harmonic maps for every (l, m), l in ells: 2l+1 maps per l.
    Normalization is irrelevant for fitting."""
    lmax = max(ells)
    out, names = [], []
    for ell in ells:
        for m in range(ell + 1):
            for part in ((1.0,) if m == 0 else (1.0, 1j)):
                a = np.zeros(hp.Alm.getsize(lmax), complex)
                a[hp.Alm.getidx(lmax, ell, m)] = part
                out.append(hp.alm2map(a, nside, lmax=lmax, pixwin=False))
                names.append(f"Y{ell}{m}{'c' if part == 1.0 else 's'}")
    return np.array(out), names


# ---------------------------------------------------------------- pixel window
def pixel_window(nside=NSIDE, lmax=3 * NSIDE - 1, nside_sub=512):
    """HEALPix pixel window w_l, computed directly (the tabulated file is not
    reachable offline). For one pixel, (1/N^2) sum_ij P_l(n_i . n_j) over
    N = (nside_sub/nside)^2 sub-pixel centres equals the pixel's |b_l|^2 x
    4pi/(2l+1); averaging this over all pixels (one per ring, weighted by the
    ring's size) gives w_l^2."""
    k = nside_sub // nside
    tot = np.zeros(lmax + 1)
    for ring in range(1, 4 * nside):
        start, npix_ring = hp.ringinfo(nside, np.array([ring]))[:2]
        p_nest = hp.ring2nest(nside, int(start[0]))
        sub = p_nest * k * k + np.arange(k * k)
        v = np.array(hp.pix2vec(nside_sub, sub, nest=True)).T
        x = np.clip((v @ v.T).ravel(), -1, 1)
        p0, p1 = np.ones_like(x), x
        s = np.empty(lmax + 1)
        s[0], s[1] = 1.0, p1.mean()
        for l in range(2, lmax + 1):
            p0, p1 = p1, ((2 * l - 1) * x * p1 - (l - 1) * p0) / l
            s[l] = p1.mean()
        tot += npix_ring[0] * s
    return np.sqrt(tot / hp.nside2npix(nside))
