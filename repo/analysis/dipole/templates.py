"""
Ecliptic-correction shapes (project Step 6).

fit_general_mult  dipole fit with any number of multiplicative correction
                  templates and optional extra multipoles:
                      n_p / nbar = a (1 + sum_k s_k t_k(p)) (1 + d . n_p + q . Y(p))
                  For fixed s the model is linear in (a-1, a s_k, a d, a q);
                  s is iterated to convergence (as fit_dipole_lsq_mult).
deviance          Poisson deviance of a fitted model (smaller = better fit)
template_maps     the candidate shapes as sky maps
"""
import numpy as np

from .geometry import pixel_vectors, NPIX
from .estimators import direction


def fit_general_mult(counts, mask, trends=None, extra=None, n_iter=50, tol=1e-12):
    seen = ~mask
    n = counts[seen].astype(float)
    nbar = n.mean()
    vec = pixel_vectors()[:, seen]
    Y = np.zeros((0, seen.sum())) if extra is None else np.asarray(extra, float)[:, seen]
    T = np.zeros((0, seen.sum())) if not trends else np.array([np.asarray(t, float)[seen] for t in trends])
    K, nq = T.shape[0], Y.shape[0]
    scale = T.std(axis=1) if K else np.zeros(0)
    s = np.zeros(K)
    for _ in range(n_iter if K else 1):
        mod = 1 + s @ T if K else np.ones(seen.sum())
        A = np.vstack([vec * mod, Y * mod, T, np.ones(seen.sum())])
        F = nbar * (A @ A.T)
        x = np.linalg.solve(F, A @ (n - nbar))
        a = 1 + x[-1]
        if not K:
            break
        s_new = x[3 + nq:3 + nq + K] / a
        done = np.all(np.abs((s_new - s) * scale) < tol)
        s = s_new
        if done:
            break
    cov = np.linalg.inv(F)
    d = x[:3] / a
    l, b = direction(d)
    q = x[3:3 + nq] / a
    model = np.zeros(NPIX)
    mod = 1 + s @ T if K else np.ones(seen.sum())
    model[seen] = a * nbar * mod * (1 + d @ vec + (q @ Y if nq else 0))
    return dict(d=d, amp=float(np.linalg.norm(d)), l=l, b=b, s=s, a=a, q=q, nbar=nbar,
                sigma=np.sqrt(np.diag(cov[:3, :3])) / a, model=model)


def deviance(counts, model, mask):
    """Poisson deviance 2 sum[n ln(n/lam) - (n - lam)] over unmasked pixels."""
    seen = ~mask
    n, lam = counts[seen].astype(float), model[seen]
    term = np.where(n > 0, n * np.log(np.where(n > 0, n, 1) / lam), 0.0)
    return float(2 * np.sum(term - (n - lam)))


def template_maps(elat, w1cov_pix):
    """Candidate correction shapes. elat: signed ecliptic latitude (deg) per
    pixel; w1cov_pix: mean W1 coverage (number of exposures) per pixel."""
    ae = np.abs(elat)
    cov = np.where(w1cov_pix > 0, w1cov_pix, np.nan)
    single = {
        "line": ae,
        "quad": ae ** 2,
        "cos": 1 - np.cos(np.radians(ae)),
        "step20": (ae > 20).astype(float),
        "cov": np.nan_to_num(cov ** -0.5),
        "north_line": ae * (elat > 0),
        "south_line": ae * (elat <= 0),
    }
    sets = {
        "none": [],
        "line": ["line"],
        "quad": ["quad"],
        "cos": ["cos"],
        "step20": ["step20"],
        "line+quad": ["line", "quad"],
        "cov": ["cov"],
        "line+cov": ["line", "cov"],
        "NS line": ["north_line", "south_line"],
    }
    return single, sets
