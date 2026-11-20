"""Amplitude likelihood, G&H (2012) Eqs. 22-23 (prototype Step 6)."""
import numpy as np

from .geometry import pixel_vectors


def amplitude_likelihood(d, cov, A_max=None, nA=1500, nside_dir=16):
    """
    L(A) ~ sum over directions n of exp[-1/2 (A n - d)^T Cov^-1 (A n - d)]

    d = fitted dipole vector, cov = its 3x3 covariance. Directions are the
    centres of an equal-area HEALPix grid (nside_dir), so the sum is the
    integral over the sphere up to a constant. Grid choice (16, 1500) agrees
    with (32, 6000) to < 0.1% (prototype Step 6).

    Returns the A grid, L(A) normalized to peak 1, and summaries: peak
    (maximum likelihood; the reported amplitude), mean and median (flat prior
    on A >= 0), and the 16-84% interval.

    Note: at our signal-to-noise the peak is biased high by ~1-2% because the
    errors differ between directions (Galactic mask); with equal errors it is
    unbiased. The dipole VECTOR is unbiased.
    """
    d = np.asarray(d, float)
    M = np.linalg.inv(cov)
    n = pixel_vectors(nside_dir)
    q1 = np.einsum("in,ij,jn->n", n, M, n)
    q2 = n.T @ (M @ d)
    c = d @ M @ d
    if A_max is None:
        A_max = np.linalg.norm(d) + 6 * np.sqrt(np.linalg.eigvalsh(cov).max())
    A = np.linspace(0, A_max, nA)
    expo = -0.5 * (A[:, None] ** 2 * q1[None] - 2 * A[:, None] * q2[None] + c)
    mx = expo.max(axis=1, keepdims=True)
    logL = (mx[:, 0] + np.log(np.exp(expo - mx).sum(axis=1)))
    L = np.exp(logL - logL.max())
    dA = A[1] - A[0]
    P = L / (L.sum() * dA)
    cdf = np.cumsum(P) * dA
    q = lambda f: float(np.interp(f, cdf, A))
    return dict(A=A, L=L, peak=float(A[np.argmax(L)]), mean=float((A * P).sum() * dA),
                median=q(0.5), lo=q(0.16), hi=q(0.84))
