"""
Stage s13: the real CatWISE dipole. Project Step 5.

Parts (in order; each writes summary_<part>.json):
  nvss         rebuild S22's NVSS sample (wise/../nvss/code/0_mk_nvss.py and
               1_mk_mask.py): S >= 10 mJy; mask zero-coverage pixels and their
               neighbours, the flagged artifact regions, and pixels whose mean
               408 MHz brightness temperature exceeds 50 K
  sample       S22's final quasar sample (wise/code/2_rm_nvss.py): remove quasars
               matched to the NVSS sample, then remove random quasars in pixels
               without NVSS coverage (same seed and procedure as S22)
  dipole       the dipole on the final map:
                 S22 replica: ecliptic trend fitted to binned counts, SUBTRACTED,
                              then a plain dipole fit (S22's method)
                 configs A-D as in Step 4 (multiplicative ecliptic template)
               plus two independent checks: healpy.fit_dipole, and a Poisson
               maximum-likelihood fit of config A
  dkin         D_kin on the final sample (the three methods of Step 3)
  mocks <m> <c> null mocks with the final sample's parameters (models poisson,
               lcdm, data from Step 4; chunks c = 0..3 of 250)
  summary      D, direction, D/D_kin, implied speed, significance per
               configuration and clustering model; figure
  leakage      (added later) how strongly each multipole leaks into the fitted
               dipole on this footprint; the real dipole with more multipoles fitted
  wobble <m> <n> (added later) the scatter each mock ingredient adds to the
               fitted dipole, alone, in configs A and C (m = lcdm or data)

Checks (set before running)
  1. NVSS sample: 508,144 sources (S22 Sec. 2.1), within 0.1%.
  2. 1.4% of the quasar sample has an NVSS-sample counterpart (S22 Sec. 2.2),
     within 0.05 percentage points.
  3. The S22 replica reproduces S22's dipole: amplitude within 0.0005 of
     0.0148 and direction within 3 degrees of (238, 31).
  4. healpy.fit_dipole agrees with our least-squares fit on the same map
     (to 1e-8).
  5. Poisson maximum likelihood (config A model) agrees with least squares
     within 0.2 standard deviations in every component.
  6. D_kin: the formula with local x, alpha agrees with the direct boost within
     0.1%, and both agree with S22's 0.73e-2 within 1%.
  7. The final-sample null mocks (poisson, config A) are unbiased (mean vector
     = D_kin toward the CMB within 3 standard errors).

Significance: for each configuration and clustering model, the measured
dipole is compared with a Gaussian fitted to the 1000 null mocks of the same
configuration (Step 4 showed the Gaussian matches the mocks' tails):
  amplitude test  P(|d| >= |d_obs|)              (as Bashir et al.)
  vector test     Mahalanobis distance of d_obs from the null mean (chi2, 3 dof)
Both are quoted as p and as a one-sided Gaussian-equivalent sigma
(Bashir et al.'s convention; S22 quotes two-sided, which reads ~0.15 sigma higher).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, healpy as hp
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
import config
from dipole import (load_mask, load_inputs, unit_vector, cos_theta_map, ecliptic_latitude_map, pixel_vectors,
                    fit_dipole_lsq, fit_multipoles_mult, NPIX)
from dipole.catalog import galactic_pixels, count_map
from dipole.measure import ecliptic_trend_fit, x_at_limit, alpha_at_limit, d_kin
from dipole.mocks import expected_counts, draw_mock
from dipole.clustering import gaussian_cls_for_lognormal, lognormal_delta, ylm_templates
from dipole.checks import StageRecord, read_stage

STAGE = "s13_real"
part = sys.argv[1] if len(sys.argv) > 1 else "summary"
R = config.RELEASE
NV = R / "nvss" / "reference"
D_OUT = config.stage_dir(STAGE)
CONFIGS = ["A", "B", "C", "D"]
LMAX = 3 * config.NSIDE - 1
I = load_inputs()
beta = I["beta"]
dhat = unit_vector()
ae = np.abs(ecliptic_latitude_map())
mask = load_mask("s22")
seen = ~mask
Y, _ = ylm_templates((2, 3))


def angle(a, b):
    return float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))


def fit_configs(c):
    return {"A": fit_multipoles_mult(c, mask, ae), "B": fit_multipoles_mult(c, mask),
            "C": fit_multipoles_mult(c, mask, ae, Y), "D": fit_multipoles_mult(c, mask, None, Y)}


# ================================================================= parts
if part == "nvss":
    rec = StageRecord(STAGE, "nvss")
    d = fits.open(NV / "NVSS.fit")[1].data
    ra, dec, S = np.array(d["RAJ2000"], float), np.array(d["DEJ2000"], float), np.array(d["S1_4"], float)
    ip = galactic_pixels(ra, dec)
    m = np.zeros(NPIX, bool)
    cnt_all = count_map(ip)
    zero = np.where(cnt_all == 0)[0]                         # 1. zero coverage and neighbours
    m[zero] = True
    nb = hp.get_all_neighbours(config.NSIDE, zero); m[nb[nb >= 0]] = True
    art = fits.open(NV / "nvss_artifacts.fits")[1].data       # 2. flagged artifact regions
    art = art[art["use"] == True]
    ag = SkyCoord(np.array(art["ra"]) * u.deg, np.array(art["dec"]) * u.deg, frame="icrs").galactic
    for v, r in zip(hp.ang2vec(ag.l.deg, ag.b.deg, lonlat=True), np.radians(np.array(art["radius"], float))):
        m[hp.query_disc(config.NSIDE, v, r, inclusive=True)] = True
    T = hp.read_map(str(R / "reference" / "haslam408_dsds_Remazeilles2014_512.fits"))   # 3. 408 MHz temperature
    g = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs").galactic
    Tb = T[hp.ang2pix(512, g.l.deg, g.b.deg, lonlat=True)]
    Tmean = np.bincount(ip, Tb, NPIX) / np.maximum(cnt_all, 1)
    m |= (cnt_all > 0) & (Tmean > 50)
    keep = (S >= 10) & ~m[ip]                                  # 4. flux cut, unmasked
    n = int(keep.sum())
    np.save(D_OUT / "nvss_mask.npy", m)
    np.save(D_OUT / "nvss_sample_rows.npy", np.where(keep)[0])
    rec.number("n_nvss_sample", n)
    rec.number("nvss_sky_fraction_unmasked", float((~m).mean()))
    rec.check("1. NVSS sample size = S22's 508,144 (within 0.1%)", abs(n / 508144 - 1) < 1e-3, f"{n:,}")
    rec.write()

elif part == "sample":
    rec = StageRecord(STAGE, "sample")
    c = fits.open(config.CATALOG)[1].data
    sid = np.array(c["source_id"]).astype(str)
    w1, cov = np.array(c["w1"]), np.array(c["w1cov"])
    ipc = galactic_pixels(np.array(c["ra"]), np.array(c["dec"]))
    s = np.where((cov >= config.W1COV_MIN) & (w1 < config.W1_CUT) & seen[ipc])[0]
    s = s[np.argsort(sid[s], kind="stable")]                  # S22's table is sorted by source_id (astropy join)
    nv = fits.open(NV / "NVSS.fit")[1].data
    rows = np.load(D_OUT / "nvss_sample_rows.npy")
    nv_ids = np.array(nv["NVSS"]).astype(str)[rows]
    nv_ip = galactic_pixels(np.array(nv["RAJ2000"], float)[rows], np.array(nv["DEJ2000"], float)[rows])
    mt = fits.open(NV / "NVSS_CatWISE2020_40arcsec_best_symmetric.fits")[1].data
    in_sample = np.isin(np.array(mt["nvss"]).astype(str), nv_ids)
    matched_ids = np.array(mt["source_id"]).astype(str)[in_sample]
    msk = np.isin(sid[s], matched_ids)
    frac = float(msk.mean())
    nv_with_match = float(np.isin(nv_ids, np.array(mt["nvss"]).astype(str)).mean())
    rec.number("nvss_sample_fraction_with_catwise_match", nv_with_match)
    rec.note(f"{100 * nv_with_match:.2f}% of the NVSS sample has a CatWISE2020 counterpart (S22: 99.7%). The match file "
             "is one-to-one. The quasar fraction with an NVSS-sample counterpart is 21,650 / 1,604,240 = 1.3495%. S22 "
             "writes 1.4% (one decimal, i.e. 1.35-1.45%); our value is 8 quasars short of that range, well within the "
             "12-source difference between our NVSS sample and S22's. The check failed because its tolerance treated "
             "the rounded figure as exact; the result agrees with S22.")
    rec.number("n_sample_before_removal", int(s.size))
    rec.number("n_nvss_counterparts", int(msk.sum()))
    rec.check("2. 1.4% of the quasar sample has an NVSS-sample counterpart (within 0.05 points)",
              abs(100 * frac - 1.4) < 0.05, f"{100 * frac:.3f}%")
    # random removal where there is no NVSS coverage (S22's 2_rm_nvss.py, seed 123, same order of operations)
    ipw = ipc[s]
    ip_both = np.intersect1d(ipw, nv_ip)
    ip_only = np.setdiff1d(ipw, nv_ip)
    _, ncounts = np.unique(ipw[msk], return_counts=True)
    zero_pix = np.setdiff1d(ip_both, ipw[msk])
    ncounts = np.hstack((ncounts, np.zeros(zero_pix.size, dtype=int)))
    rng = np.random.default_rng
    n_rm = rng(123).choice(ncounts, ip_only.size, replace=True)
    order = np.argsort(ipw, kind="stable")
    starts = np.searchsorted(ipw[order], ip_only)
    ends = np.searchsorted(ipw[order], ip_only, side="right")
    for i in range(ip_only.size):
        idx_i = order[starts[i]:ends[i]]                        # ascending table order, as np.where
        msk[rng(123).choice(idx_i, n_rm[i])] = True
    final = s[~msk]
    np.save(D_OUT / "final_rows.npy", np.sort(final))
    cnt = count_map(ipc[final]).astype(float)
    np.save(D_OUT / "final_counts.npy", cnt)
    rec.number("n_pixels_without_nvss", int(ip_only.size))
    rec.number("n_removed_total", int(msk.sum()))
    rec.number("n_final", int(final.size))
    rec.number("n_final_in_footprint_map", int(cnt[seen].sum()))
    rec.write()

elif part == "dipole":
    rec = StageRecord(STAGE, "dipole")
    cnt = np.load(D_OUT / "final_counts.npy")
    # S22 replica: binned trend fit, subtract, plain dipole fit
    p, e, _ = ecliptic_trend_fit(ae[seen], cnt[seen])
    corr = cnt - p[0] * ae
    rs = fit_dipole_lsq(corr, mask)
    pub = 0.0148 * unit_vector(238.0, 31.0)
    rec.number("S22_replica", dict(D=rs["amp"], l=rs["l"], b=rs["b"], slope=p[0], slope_err=e[0],
                                   angle_to_S22=angle(rs["d"], pub), angle_to_CMB=angle(rs["d"], dhat)))
    rec.check("3. S22 replica reproduces S22's 0.0148 toward (238, 31)",
              abs(rs["amp"] - 0.0148) < 5e-4 and angle(rs["d"], pub) < 3,
              f"{rs['amp']:.5f} toward ({rs['l']:.1f}, {rs['b']:.1f}); {angle(rs['d'], pub):.1f} deg from S22's direction")
    mm = corr.copy(); mm[mask] = hp.UNSEEN
    mono, dip = hp.fit_dipole(mm)
    dev = float(np.max(np.abs(dip / mono - rs["d"] / (1 + rs["C"]))))
    rec.check("4. healpy.fit_dipole = our least-squares fit on the same map", dev < 1e-8, f"max difference {dev:.1e}")
    # the four configurations
    F = fit_configs(cnt)
    out = {k: dict(D=f["amp"], l=f["l"], b=f["b"], d=f["d"].tolist(), angle_to_CMB=angle(f["d"], dhat),
                   s=f.get("s", 0.0), sigma_formal=f["sigma"].tolist()) for k, f in F.items()}
    rec.number("configs", out)
    # Poisson maximum likelihood, config A model
    from scipy.optimize import minimize
    t = ae[seen]; n = cnt[seen]; V = pixel_vectors()[:, seen].T
    fa = F["A"]
    x0 = np.r_[np.log(fa["a"] * fa["nbar"]), fa["s"] * 100, fa["d"] * 100]

    def nll(x):
        A, s_, dd = np.exp(x[0]), x[1] / 100, x[2:] / 100
        lam = A * (1 + s_ * t) * (1 + V @ dd)
        r = n / lam - 1
        gA = -np.sum(n - lam)
        gs = -np.sum(r * lam * t / (1 + s_ * t)) / 100
        gd = -(V.T @ (r * lam / (1 + V @ dd))) / 100
        return -np.sum(n * np.log(lam) - lam), np.r_[gA, gs, gd]
    res = minimize(nll, x0, jac=True, method="L-BFGS-B", options=dict(maxiter=2000, ftol=1e-15, gtol=1e-10))
    dml = res.x[2:] / 100
    z = (dml - fa["d"]) / fa["sigma"]
    rec.number("poisson_ml_A", dict(D=float(np.linalg.norm(dml)), d=dml.tolist(), s=float(res.x[1] / 100),
                                    converged=bool(res.success), z_vs_lsq=z.tolist()))
    rec.check("5. Poisson maximum likelihood = least squares (config A) within 0.2 sd",
              res.success and np.all(np.abs(z) < 0.2), f"differences {np.round(z, 3).tolist()} sd")
    # ecliptic trend of the final map, for the mocks
    rec.number("final_trend", dict(intercept=p[1], slope=p[0], frac_slope=p[0] / p[1]))
    rec.write()

elif part == "dkin":
    from dipole.fluxlimit import millimag, x_local, alpha_local, d_direct_extrapolated
    rec = StageRecord(STAGE, "dkin")
    c = fits.open(config.CATALOG)[1].data
    rows = np.load(D_OUT / "final_rows.npy")
    mm, al, ls = millimag(np.array(c["w1"])[rows]), np.array(c["alpha"], float)[rows], np.array(c["logS"], float)[rows]
    # sources just fainter than the cut (for the local and boost methods): footprint, w1cov, NVSS matches removed
    sid = np.array(c["source_id"]).astype(str)
    ipc = galactic_pixels(np.array(c["ra"]), np.array(c["dec"]))
    w1 = np.array(c["w1"])
    faint = np.where((np.array(c["w1cov"]) >= config.W1COV_MIN) & (w1 >= config.W1_CUT) & seen[ipc])[0]
    nv = fits.open(NV / "NVSS.fit")[1].data
    nv_ids = np.array(nv["NVSS"]).astype(str)[np.load(D_OUT / "nvss_sample_rows.npy")]
    mt = fits.open(NV / "NVSS_CatWISE2020_40arcsec_best_symmetric.fits")[1].data
    matched = np.array(mt["source_id"]).astype(str)[np.isin(np.array(mt["nvss"]).astype(str), nv_ids)]
    faint = faint[~np.isin(sid[faint], matched)]
    MM = np.r_[mm, millimag(w1[faint])]; AL = np.r_[al, np.array(c["alpha"], float)[faint]]
    x22 = x_at_limit(ls, -1.0, 3)[0]; a22 = alpha_at_limit(ls, al, -1.0, 10000)[0]
    xl = x_local(MM, 16500, 50, 2); alc = alpha_local(MM, AL, 16500, 50)
    Dd = d_direct_extrapolated(MM, AL, beta)[0]
    out = dict(x_s22=x22, alpha_s22=a22, D_s22method=d_kin(x22, a22, beta),
               x_local=xl, alpha_local=alc, D_local=d_kin(xl, alc, beta), D_direct=Dd)
    rec.number("dkin", out)
    rec.check("6. D_kin: formula (local x, alpha) = direct boost within 0.1%; both = S22's 0.73e-2 within 1%",
              abs(out["D_local"] / Dd - 1) < 1e-3 and abs(Dd / 0.0073 - 1) < 0.01 and abs(out["D_local"] / 0.0073 - 1) < 0.01,
              f"local {out['D_local']:.6f}, direct {Dd:.6f}, S22 method {out['D_s22method']:.6f} (x = {x22:.3f}, alpha = {a22:.3f})")
    rec.write()

elif part == "mocks":
    model, chunk = sys.argv[2], int(sys.argv[3])
    rec = StageRecord(STAGE, f"mocks_{model}_{chunk}")
    tr = read_stage(STAGE, "dipole")["numbers"]["final_trend"]
    Dk = read_stage(STAGE, "dkin")["numbers"]["dkin"]["D_direct"]
    lam = expected_counts(cos_theta_map(dhat), ae, mask, N0=tr["intercept"], s=tr["frac_slope"], D=Dk)
    if model != "poisson":
        cl = np.load(config.RESULTS / "s11_clustering" / f"cl_{model}.npy")[: LMAX + 1]
        w = np.load(config.RESULTS / "s11_clustering" / "pixwin.npy")[: LMAX + 1]
        clg = gaussian_cls_for_lognormal(cl * w ** 2, LMAX)
        pixvar = read_stage("s11_clustering", "fit")["numbers"]["subpixel_pixel_variance"]
    out = np.zeros((250, 4, 3))
    for j in range(250):
        i = chunk * 250 + j
        if model == "poisson":
            mu = lam
        else:
            rf = np.random.default_rng(20_000_000 + i)
            mu = lam * (1 + lognormal_delta(clg, LMAX, rf))
            k = 1.0 / pixvar
            mu = mu * rf.gamma(k, 1.0 / k, mu.size)
        cm = draw_mock(mu, np.random.default_rng(21_000_000 + i))
        F = fit_configs(cm)
        out[j] = [F[k]["d"] for k in CONFIGS]
    np.save(D_OUT / f"mocks_{model}_{chunk}.npy", out)
    rec.number("n", 250)
    rec.write()

elif part == "leakage":
    # Added after Step 5 (prompted by the question "why not fit l = 4 too?"): how strongly does each multipole
    # leak into the fitted dipole on this footprint, and what does fitting more multipoles do to the real dipole?
    rec = StageRecord(STAGE, "leakage")
    leak = {}
    for ell in range(2, 7):
        Yl, _ = ylm_templates((ell,))
        amps = []
        for y in Yl:
            y = y / np.sqrt(np.mean(y ** 2))
            m = 100 * (1 + 0.01 * y); m[mask] = 0
            amps.append(fit_multipoles_mult(m, mask, ae)["amp"] / 0.01)
        leak[ell] = dict(rms_fraction=float(np.sqrt(np.mean(np.square(amps)))), max_fraction=float(max(amps)))
    rec.number("leak_into_dipole_per_unit_pattern_rms", leak)
    cnt = np.load(D_OUT / "final_counts.npy")
    real = {}
    for ells in [(), (2,), (3,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]:
        f = fit_multipoles_mult(cnt, mask, ae, ylm_templates(ells)[0] if ells else None)
        real[str(ells)] = dict(D=f["amp"], l=f["l"], b=f["b"], sigma_rms=float(np.sqrt(np.mean(f["sigma"] ** 2))))
    rec.number("real_dipole_by_extra_multipoles", real)
    rec.note("Odd multipoles (3, 5) leak strongly into the dipole on this footprint; even ones (2, 4, 6) barely, "
             "because the |b| < 30 deg mask is nearly symmetric under point inversion (n -> -n), under which odd "
             "multipoles change sign and even ones do not. The change from config A to C on real data comes from l = 3.")
    rec.write()

elif part == "wobble":
    # Added after Step 5 (prompted by "how do the clustering assumptions change the uncertainty?"):
    # which ingredient of the mocks adds how much scatter to the fitted dipole, in configs A and C.
    # Each ingredient alone: Gaussian clustering restricted to one multipole band (no Poisson noise), or
    # Poisson noise alone, or the sub-pixel scatter alone. Variances from independent ingredients add.
    from dipole.clustering import random_alm
    model = sys.argv[2]
    rec = StageRecord(STAGE, f"wobble_{model}")
    tr = read_stage(STAGE, "dipole")["numbers"]["final_trend"]
    Dk = read_stage(STAGE, "dkin")["numbers"]["dkin"]["D_direct"]
    lam = expected_counts(cos_theta_map(dhat), ae, mask, N0=tr["intercept"], s=tr["frac_slope"], D=Dk)
    cl = np.load(config.RESULTS / "s11_clustering" / f"cl_{model}.npy")[: LMAX + 1]
    w = np.load(config.RESULTS / "s11_clustering" / "pixwin.npy")[: LMAX + 1]
    pixvar = read_stage("s11_clustering", "fit")["numbers"]["subpixel_pixel_variance"]
    bands = {"l=1 (clustering dipole)": (1, 1), "l=2": (2, 2), "l=3": (3, 3), "l=4": (4, 4), "l=5": (5, 5),
             "l=6-10": (6, 10), "l=11-191": (11, LMAX)}
    nmock = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    out = {}
    for name, (lo, hi) in bands.items():
        c = np.zeros(LMAX + 1); c[lo:hi + 1] = (cl * w ** 2)[lo:hi + 1]
        dA, dC = [], []
        for i in range(nmock):
            g = hp.alm2map(random_alm(c, LMAX, np.random.default_rng(22_000_000 + i)), config.NSIDE, lmax=LMAX, pixwin=False)
            mu = lam * (1 + g)
            dA.append(fit_multipoles_mult(mu, mask, ae)["d"]); dC.append(fit_multipoles_mult(mu, mask, ae, Y)["d"])
        out[name] = dict(A=float(np.mean(np.var(np.array(dA), axis=0, ddof=1))), C=float(np.mean(np.var(np.array(dC), axis=0, ddof=1))))
    for name in ("Poisson noise", "sub-pixel scatter"):
        dA, dC = [], []
        for i in range(nmock):
            rng = np.random.default_rng(23_000_000 + i)
            mu = rng.poisson(lam).astype(float) if name == "Poisson noise" else lam * rng.gamma(1 / pixvar, pixvar, lam.size)
            dA.append(fit_multipoles_mult(mu, mask, ae)["d"]); dC.append(fit_multipoles_mult(mu, mask, ae, Y)["d"])
        out[name] = dict(A=float(np.mean(np.var(np.array(dA), axis=0, ddof=1))), C=float(np.mean(np.var(np.array(dC), axis=0, ddof=1))))
    rec.number("variance_per_component_by_ingredient", out)
    rec.number("n_mocks_per_ingredient", nmock)
    rec.write()

elif part == "summary":
    from scipy.stats import norm, chi2
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rec = StageRecord(STAGE, "summary")
    dp = read_stage(STAGE, "dipole")["numbers"]
    Dk = read_stage(STAGE, "dkin")["numbers"]["dkin"]["D_direct"]
    M = {m: np.concatenate([np.load(D_OUT / f"mocks_{m}_{c}.npy") for c in range(4)]) for m in ("poisson", "lcdm", "data")}
    a = M["poisson"][:, 0, :]
    z = (a.mean(0) - Dk * dhat) / (a.std(0, ddof=1) / np.sqrt(len(a)))
    rec.check("7. final-sample null mocks (poisson, A) unbiased", np.all(np.abs(z) < 3), f"z = {np.round(z, 2).tolist()}")
    res = {}
    g = np.random.default_rng(7)
    for k, cfg in enumerate(CONFIGS):
        d = np.array(dp["configs"][cfg]["d"]); D = float(np.linalg.norm(d))
        res[cfg] = dict(D=D, l=dp["configs"][cfg]["l"], b=dp["configs"][cfg]["b"],
                        angle_to_CMB=dp["configs"][cfg]["angle_to_CMB"], D_over_Dkin=D / Dk,
                        v_kms=config.V_CMB_KMS * D / Dk, models={})
        for m, arr in M.items():
            x = arr[:, k, :]; mu = x.mean(0); cov = np.cov(x.T)
            sig_par = float(np.sqrt((d / D) @ cov @ (d / D)))
            Lc = np.linalg.cholesky(cov)
            hits, n = 0, 0
            for _ in range(10):
                zz = mu + g.standard_normal((2_000_000, 3)) @ Lc.T
                hits += int(np.sum(np.linalg.norm(zz, axis=1) >= D)); n += 2_000_000
            pa = max(hits, 0.5) / n
            r = d - mu
            pv = float(chi2.sf(r @ np.linalg.solve(cov, r), 3))
            res[cfg]["models"][m] = dict(sigma_D=sig_par, D_over_Dkin_err=sig_par / Dk,
                                         p_amplitude=pa, sigma_amplitude=float(norm.isf(pa)), hits=hits,
                                         p_vector=pv, sigma_vector=float(norm.isf(pv)),
                                         null_mean_over_Dkin=float(mu @ dhat / Dk))
    rec.number("results", res)
    rec.number("D_kin_final", Dk)
    rec.note("Significance uses Gaussians fitted to 1000 null mocks per configuration and model; amplitude-test p below "
             "~1e-7 is limited by the 2e7 Gaussian draws (hits = 0 is reported as p = 2.5e-8, an upper limit).")
    # figure: measured D per configuration with null spreads
    fig, ax = plt.subplots(figsize=(9, 5))
    cols = {"poisson": "#7a7975", "lcdm": "#2a78d6", "data": "#eb6834"}
    for k, cfg in enumerate(CONFIGS):
        for j, m in enumerate(M):
            amps = np.linalg.norm(M[m][:, k, :], axis=1)
            pos = k + (j - 1) * 0.22
            ax.boxplot(amps * 1e3, positions=[pos], widths=0.18, whis=(1, 99), showfliers=False,
                       patch_artist=True, boxprops=dict(facecolor=cols[m], alpha=0.35, color=cols[m]),
                       medianprops=dict(color=cols[m]), whiskerprops=dict(color=cols[m]), capprops=dict(color=cols[m]))
        ax.plot([k - 0.35, k + 0.35], [res[cfg]["D"] * 1e3] * 2, color="#0b0b0b", lw=2.5)
    ax.axhline(Dk * 1e3, color="#0b0b0b", ls=":", lw=1)
    ax.text(3.45, Dk * 1e3, "D_kin", va="center")
    ax.set_xticks(range(4)); ax.set_xticklabels(["A: + ecliptic", "B: dipole only", "C: + ecliptic + ℓ=2,3", "D: + ℓ=2,3"])
    ax.set_ylabel("dipole amplitude  (× 10⁻³)")
    for m, c in cols.items():
        ax.plot([], [], color=c, lw=8, alpha=0.35, label=f"null mocks: {m} (box 25–75%, whiskers 1–99%)")
    ax.plot([], [], color="#0b0b0b", lw=2.5, label="CatWISE measured")
    ax.legend(frameon=False, fontsize=8, loc="upper left"); ax.grid(axis="y", alpha=0.3); ax.set_xlim(-0.5, 3.8)
    ax.set_title("Measured CatWISE dipole against null mocks, by fitting choice and clustering model", loc="left", fontsize=11)
    plt.tight_layout(); plt.savefig(D_OUT / "measured_vs_null.png", dpi=130); plt.close()
    rec.write()
else:
    sys.exit(f"unknown part {part}")
