"""
Measurements from catalog / maps (prototype Step 2):
  * the ecliptic-latitude density trend, fitted as in S22's code;
  * x and alpha at the flux limit, computed as in S22's code.
"""
import numpy as np


def ecliptic_trend_fit(abselat, y, binsize=200):
    """Sort pixels by |ecliptic latitude|, take means of `binsize` pixels at
    a time, fit a straight line through the bin means weighted by their
    standard errors (S22's xyfit/binxy). Returns (slope, intercept),
    their errors, and the binned points."""
    idx = np.argsort(abselat)
    x, y = abselat[idx], y[idx]
    edges = np.arange(0, x.size, binsize)
    pairs = list(zip(edges[:-1], edges[1:]))
    xc = np.array([x[a:b].mean() for a, b in pairs])
    yc = np.array([y[a:b].mean() for a, b in pairs])
    se = np.array([y[a:b].std(ddof=1) / np.sqrt(binsize) for a, b in pairs])
    p, pcov = np.polyfit(xc, yc, 1, w=1 / se, cov=True)
    return p, np.sqrt(np.diag(pcov)), (xc, yc, se)


def alpha_at_limit(logS, alpha, logS_max=-1.0, binsize=10000):
    """Mean alpha in bins of `binsize` sources (sorted by flux), straight line
    through bin means with log S < logS_max, evaluated at the faintest flux
    (S22's alpha_at_S). Returns alpha_lim and the binned curve."""
    logS_min = logS.min()
    o = np.argsort(logS)
    ss, aa = logS[o], alpha[o]
    e = np.arange(0, ss.size, binsize)
    sc = np.array([ss[a:b].mean() for a, b in zip(e[:-1], e[1:])])
    ac = np.array([aa[a:b].mean() for a, b in zip(e[:-1], e[1:])])
    k = sc < logS_max
    p = np.polyfit(sc[k], ac[k], 1)
    return float(np.polyval(p, logS_min)), (sc, ac, p, k)


def x_at_limit(logS, logS_max=-1.0, deg=3):
    """Cumulative counts N(>=S); polynomial of degree `deg` in log N vs log S
    for log S < logS_max; x = -dlogN/dlogS at the faintest flux (S22's
    x_at_S). Returns x and the curve."""
    logS_min = logS.min()
    ss = np.sort(logS)
    uS, first = np.unique(ss, return_index=True)
    logN = np.log10(ss.size - first)
    k = uS < logS_max
    p = np.polyfit(uS[k], logN[k], deg)
    return float(-np.polyval(np.polyder(p), logS_min)), (uS, logN, p, k)


def d_kin(x, alpha, beta):
    """Ellis & Baldwin (1984): D = [2 + x(1 + alpha)] beta."""
    return (2 + x * (1 + alpha)) * beta
