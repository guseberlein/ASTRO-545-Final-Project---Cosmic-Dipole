"""
Dipole estimators (prototype Step 5).

fit_dipole_lsq       Gibelyou & Huterer (2012) / Hirata (2009) linear
                     estimator, templates ADDED to the density (G&H Eq. 11).
fit_dipole_lsq_mult  Our variant for a systematic that MULTIPLIES the density
                     (used for the main results; see AI_log / NOTES: the
                     additive form is biased by ~1.5% for such a systematic).
"""
import numpy as np
import healpy as hp

from .geometry import pixel_vectors


def direction(d):
    """Galactic (l, b) in degrees of vector d; (nan, nan) if d is zero."""
    a = np.linalg.norm(d)
    if a == 0:
        return float("nan"), float("nan")
    l, b = hp.vec2ang(d / a, lonlat=True)
    return float(l[0]), float(b[0])


def fit_dipole_lsq(counts, mask, templates=None):
    """
    Model for every unmasked pixel p:
        delta_p = n_p / nbar - 1 = d . n_p + sum_i k_i t_i(p) + C
    nbar = mean count over unmasked pixels, n_p = pixel unit vector,
    t_i = template maps (dict name -> full-sky array), C = constant offset.

    G&H's 'random catalog' becomes uniform coverage of the unmasked pixels,
    so their Eqs. 16-17 read
        g_i  = sum_p T_i(p) (n_p - nbar)
        F_ij = nbar * sum_p T_i(p) T_j(p),   T = (n_x, n_y, n_z, t_1..t_N, 1);
    best fit x = F^-1 g (Eq. 13), covariance F^-1 (Sec. III B).
    """
    seen = ~mask
    n = counts[seen].astype(float)
    nbar = n.mean()
    rows = list(pixel_vectors()[:, seen])
    names = ["dx", "dy", "dz"]
    for name, t in (templates or {}).items():
        rows.append(np.asarray(t, float)[seen])
        names.append(name)
    rows.append(np.ones(seen.sum()))
    names.append("C")
    T = np.array(rows)
    F = nbar * (T @ T.T)
    g = T @ (n - nbar)
    x = np.linalg.solve(F, g)
    cov = np.linalg.inv(F)
    d = x[:3]
    l, b = direction(d)
    return dict(d=d, amp=float(np.linalg.norm(d)), l=l, b=b,
                k={nm: float(v) for nm, v in zip(names[3:-1], x[3:-1])},
                C=float(x[-1]), cov=cov, sigma=np.sqrt(np.diag(cov)),
                names=names, nbar=nbar)


def fit_dipole_lsq_mult(counts, mask, trend, n_iter=10):
    """
    Model:  n_p / nbar = a (1 + s t_p) (1 + d . n_p)
    -- the dipole modulates the local density, whatever the trend t does.
    For fixed s the model is linear in (a-1, a s, a d):
        delta_p = (a-1) + (a s) t_p + (a d) . [(1 + s t_p) n_p]
    so the dipole templates are the pixel vectors times the current trend;
    s is updated from the fit and the fit repeated until it converges.
    Same weights and covariance formula as fit_dipole_lsq.
    """
    seen = ~mask
    n = counts[seen].astype(float)
    nbar = n.mean()
    t = np.asarray(trend, float)[seen]
    vec = pixel_vectors()[:, seen]
    s = 0.0
    for _ in range(n_iter):
        T = np.vstack([vec * (1 + s * t), t, np.ones(t.size)])
        F = nbar * (T @ T.T)
        g = T @ (n - nbar)
        x = np.linalg.solve(F, g)
        a = 1 + x[4]
        s_new = x[3] / a
        if abs(s_new - s) < 1e-12:
            s = s_new
            break
        s = s_new
    cov = np.linalg.inv(F)
    d = x[:3] / a
    l, b = direction(d)
    return dict(d=d, amp=float(np.linalg.norm(d)), l=l, b=b, s=s, a=a,
                cov_d=cov[:3, :3] / a**2, sigma=np.sqrt(np.diag(cov[:3, :3])) / a,
                nbar=nbar)


def fit_multipoles_mult(counts, mask, trend=None, extra=None, n_iter=10):
    """
    Dipole fit with optional multiplicative trend and extra multipole templates
    (project Step 4):
        n_p / nbar = a (1 + s t_p) (1 + d . n_p + sum_k q_k Y_k(p))
    trend=None fixes s = 0 (no ecliptic template); extra = array of template
    maps (e.g. the 5 + 7 real Y_lm for l = 2, 3) or None.
    For fixed s the model is linear in (a-1, a s, a d, a q); s is iterated as
    in fit_dipole_lsq_mult. With extra=None and a trend this is identical to
    fit_dipole_lsq_mult; with neither it is the plain G&H dipole fit.
    """
    seen = ~mask
    n = counts[seen].astype(float)
    nbar = n.mean()
    vec = pixel_vectors()[:, seen]
    Y = np.zeros((0, seen.sum())) if extra is None else np.asarray(extra, float)[:, seen]
    has_t = trend is not None
    t = np.asarray(trend, float)[seen] if has_t else np.zeros(seen.sum())
    s = 0.0
    nq = Y.shape[0]
    for _ in range(n_iter if has_t else 1):
        mod = 1 + s * t
        rows = [vec * mod, Y * mod] + ([t] if has_t else []) + [np.ones(t.size)]
        T = np.vstack(rows)
        F = nbar * (T @ T.T)
        g = T @ (n - nbar)
        x = np.linalg.solve(F, g)
        a = 1 + x[-1]
        if not has_t:
            break
        s_new = x[3 + nq] / a
        if abs(s_new - s) < 1e-12:
            s = s_new
            break
        s = s_new
    cov = np.linalg.inv(F)
    d = x[:3] / a
    l, b = direction(d)
    return dict(d=d, amp=float(np.linalg.norm(d)), l=l, b=b, s=s, a=a, q=x[3:3 + nq] / a,
                cov_d=cov[:3, :3] / a**2, sigma=np.sqrt(np.diag(cov[:3, :3])) / a, nbar=nbar)
