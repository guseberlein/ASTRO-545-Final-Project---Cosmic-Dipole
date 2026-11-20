"""
Stage s16: is the excess kinematic or intrinsic? Project Step 8.

Motion makes each sample's dipole proportional to its own Doppler factor
K = 2 + x (1 + alpha) (D = K beta). A real anisotropy in the sources, or a
systematic common to all of them, adds the same dipole whatever K is. So the
final CatWISE sample (Step 5) is split by spectral index into three groups
with different K, and the group dipoles are fitted as
        d_i = K_i b + a
  b  the part that scales with K: an implied velocity vector (v = c b)
  a  the part common to all groups (structure or systematics; shared
     clustering also ends up here, so b is nearly free of it)
Fast motion predicts b = (v/c) toward the motion and a = 0; an intrinsic
excess predicts b = beta_CMB toward the CMB and a = the excess.

Spectral-index noise: a group selected by MEASURED alpha has a mean TRUE
alpha closer to the sample mean (regression to the mean). The catalog has no
photometric errors; the spread of alpha does not grow toward the flux limit,
so the noise may be small. Two cases are run: R = 1 (no noise) and R = 0.7
(an assumed W1-W2 noise of ~0.1 mag, alpha noise ~0.3; R = shrinkage factor).

Parts:
  split          group edges (terciles of alpha near the flux limit, applied to
                 all magnitudes; a colour selection, unchanged by motion), count
                 maps, ecliptic trends, K_i (direct boost and formula; R = 1, 0.7)
  real           group dipoles (configs A and C) on the real sky
  mocks <h> <m> <c>  mocks with ONE shared clustering field and Poisson counts
                 per group; hypotheses h: kin (D_i = K_i beta_CMB), fast
                 (D_i = 2 K_i beta_CMB), intr (D_i = K_i beta_CMB + the full
                 sample's excess, the same for every group); clustering model m
                 (lcdm, data); chunks c = 0..2 of 100
  summary        fits of b and a, tests, null calibration, Quaia velocity, figure
  robust         (added after the summary) are the group dipoles equal? Several
                 ecliptic corrections; worst case with independent clustering
  composition    (added later) does the colour mix change across the sky? Each
                 group's share against WISE coverage, the dipole of each share
                 map, and the high-minus-low group dipole difference

Checks (set before running)
  1. The groups add up to the full final sample (counts per pixel identical).
  2. Each group's K from the direct boost equals the formula with local x,
     alpha within 1% (R = 1).
  3. Kinematic-only mocks (kin): fitted b is unbiased (= beta_CMB toward the
     CMB within 3 standard errors per component) and a has mean zero, config A,
     both clustering models.
  4. Kinematic-only mocks: the test "is a = 0 rejected at p < 0.05" fires in
     5% +/- 3% of mocks (calibrated test), config A, both models.
"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, healpy as hp
from astropy.io import fits
import config
from dipole import (load_mask, unit_vector, cos_theta_map, ecliptic_latitude_map, pixel_vectors,
                    fit_multipoles_mult, NPIX)
from dipole.catalog import galactic_pixels, count_map
from dipole.measure import ecliptic_trend_fit, d_kin
from dipole.mocks import expected_counts
from dipole.fluxlimit import millimag, x_local, alpha_local, d_direct_extrapolated
from dipole.clustering import gaussian_cls_for_lognormal, lognormal_delta, ylm_templates
from dipole.checks import StageRecord, read_stage

STAGE = "s16_origin"
part = sys.argv[1] if len(sys.argv) > 1 else "summary"
OUT = config.stage_dir(STAGE)
LMAX = 3 * config.NSIDE - 1
I = json.loads((config.RESULTS / "s02_inputs" / "inputs.json").read_text())
beta = I["beta"]
mask = load_mask("s22"); seen = ~mask
ae = np.abs(ecliptic_latitude_map())
dhat = unit_vector()
Y, _ = ylm_templates((2, 3))
G = 3
RS = (1.0, 0.7)


def fit_cfg(c, cfg):
    return fit_multipoles_mult(c, mask, ae, Y if cfg == "C" else None)


def design(K):
    """Rows: group i, component j; columns: b (3), a (3)."""
    X = np.zeros((3 * len(K), 6))
    for i, k in enumerate(K):
        X[3 * i:3 * i + 3, :3] = k * np.eye(3)
        X[3 * i:3 * i + 3, 3:] = np.eye(3)
    return X


def gls(d, X, Sigma):
    W = np.linalg.inv(Sigma)
    F = X.T @ W @ X
    p = np.linalg.solve(F, X.T @ W @ d)
    return p, np.linalg.inv(F), float((d - X @ p) @ W @ (d - X @ p))


def tests(d, K, Sigma):
    """Fit d = K b + a; test a = 0 (pure scaling: motion) and b = beta_CMB dhat
    (kinematic CMB motion plus a common excess)."""
    from scipy.stats import chi2
    X = design(K)
    p, C, c2 = gls(d, X, Sigma)
    a, Ca = p[3:], C[3:, 3:]
    b, Cb = p[:3], C[:3, :3]
    chi_a = float(a @ np.linalg.solve(Ca, a))
    r = b - beta * dhat
    chi_b = float(r @ np.linalg.solve(Cb, r))
    return dict(b=b, a=a, Cb=Cb, Ca=Ca, chi2_fit=c2, p_a0=float(chi2.sf(chi_a, 3)), p_bcmb=float(chi2.sf(chi_b, 3)),
                v_kms=float(np.linalg.norm(b) * config.C_KMS))


# ================================================================= parts
if part == "split":
    rec = StageRecord(STAGE, "split")
    c = fits.open(config.CATALOG)[1].data
    rows = np.load(config.RESULTS / "s13_real" / "final_rows.npy")
    w1, al = np.array(c["w1"], float), np.array(c["alpha"], float)
    ipc = galactic_pixels(np.array(c["ra"]), np.array(c["dec"]))
    near = rows[w1[rows] >= 16.3]
    edges = np.percentile(al[near], [100 / 3, 200 / 3])
    grp = np.digitize(al, edges)                                # 0, 1, 2 by measured alpha
    full = np.load(config.RESULTS / "s13_real" / "final_counts.npy")
    tot = np.zeros(NPIX)
    # fainter sources for the boost (as s13 part dkin): footprint, w1cov, 16.5 <= W1 < 16.6, NVSS matches removed
    sid = np.array(c["source_id"]).astype(str)
    faint = np.where((np.array(c["w1cov"]) >= config.W1COV_MIN) & (w1 >= config.W1_CUT) & seen[ipc])[0]
    nv = fits.open(config.RELEASE / "nvss" / "reference" / "NVSS.fit")[1].data
    nv_ids = np.array(nv["NVSS"]).astype(str)[np.load(config.RESULTS / "s13_real" / "nvss_sample_rows.npy")]
    mt = fits.open(config.RELEASE / "nvss" / "reference" / "NVSS_CatWISE2020_40arcsec_best_symmetric.fits")[1].data
    matched = np.array(mt["source_id"]).astype(str)[np.isin(np.array(mt["nvss"]).astype(str), nv_ids)]
    faint = faint[~np.isin(sid[faint], matched)]
    out = {"edges": edges.tolist(), "groups": []}
    ok2 = True
    for g in range(G):
        r_ = rows[grp[rows] == g]; f_ = faint[grp[faint] == g]
        cnt = count_map(ipc[r_]).astype(float); tot += cnt
        np.save(OUT / f"counts_g{g}.npy", cnt)
        p, e, _ = ecliptic_trend_fit(ae[seen], cnt[seen])
        MM = np.r_[millimag(w1[r_]), millimag(w1[f_])]; AL = np.r_[al[r_], al[f_]]
        mu_lim = float(alpha_local(millimag(w1[rows]), al[rows], 16500, 50))          # whole-sample mean at the limit
        gi = dict(n=int(r_.size), alpha_mean=float(al[r_].mean()), trend=dict(intercept=p[1], frac_slope=p[0] / p[1]))
        xl = x_local(MM, 16500, 50, 2)
        for R in RS:
            ALr = mu_lim + R * (AL - mu_lim)
            Dd = d_direct_extrapolated(MM, ALr, beta)[0]
            al_l = alpha_local(MM, ALr, 16500, 50)
            gi[f"R{R}"] = dict(x=xl, alpha_lim=al_l, K_direct=Dd / beta, K_formula=d_kin(xl, al_l, beta) / beta)
        ok2 &= abs(gi["R1.0"]["K_direct"] / gi["R1.0"]["K_formula"] - 1) < 0.01
        out["groups"].append(gi)
    rec.check("1. groups add up to the full final sample", np.array_equal(tot, full), f"{int(tot.sum()):,} sources")
    rec.check("2. K by direct boost = formula within 1% (each group, R = 1)", ok2,
              ", ".join(f"{g['R1.0']['K_direct']:.3f}/{g['R1.0']['K_formula']:.3f}" for g in out["groups"]))
    rec.number("split", out)
    rec.write()

elif part == "real":
    rec = StageRecord(STAGE, "real")
    res = {}
    for cfg in ("A", "C"):
        res[cfg] = [fit_cfg(np.load(OUT / f"counts_g{g}.npy"), cfg)["d"].tolist() for g in range(G)]
    rec.number("group_dipoles", res)
    rec.write()

elif part == "mocks":
    h, model, chunk = sys.argv[2], sys.argv[3], int(sys.argv[4])
    rec = StageRecord(STAGE, f"mocks_{h}_{model}_{chunk}")
    sp = read_stage(STAGE, "split")["numbers"]["split"]["groups"]
    K = np.array([g["R1.0"]["K_direct"] for g in sp])
    full_d = np.array(read_stage("s13_real", "dipole")["numbers"]["configs"]["A"]["d"])
    Dk_full = read_stage("s13_real", "dkin")["numbers"]["dkin"]["D_direct"]
    excess = full_d - Dk_full * dhat
    cth = cos_theta_map(dhat)
    lams = []
    for g in range(G):
        tr = sp[g]["trend"]
        base = expected_counts(cth, ae, mask, N0=tr["intercept"], s=tr["frac_slope"], D=0.0)
        dvec = {"kin": K[g] * beta * dhat, "fast": 2 * K[g] * beta * dhat, "intr": K[g] * beta * dhat + excess}[h]
        lams.append(base * (1 + pixel_vectors().T @ dvec))
    w = np.load(config.RESULTS / "s11_clustering" / "pixwin.npy")[: LMAX + 1]
    clg = gaussian_cls_for_lognormal(np.load(config.RESULTS / "s11_clustering" / f"cl_{model}.npy")[: LMAX + 1] * w ** 2, LMAX)
    pv = read_stage("s11_clustering", "fit")["numbers"]["subpixel_pixel_variance"]
    n = 100
    out = np.zeros((n, 2, G, 3))
    for j in range(n):
        i = chunk * n + j
        rf = np.random.default_rng(30_000_000 + i)                 # ONE shared field per mock
        field = (1 + lognormal_delta(clg, LMAX, rf)) * rf.gamma(1 / pv, pv, NPIX)
        for g in range(G):
            cm = np.random.default_rng(31_000_000 + 10 * i + g).poisson(lams[g] * field).astype(float)
            for k, cfg in enumerate(("A", "C")):
                out[j, k, g] = fit_cfg(cm, cfg)["d"]
    np.save(OUT / f"mocks_{h}_{model}_{chunk}.npy", out)
    rec.number("n", n)
    rec.write()

elif part == "robust":
    # Added after the summary showed the group dipoles differ far more than either hypothesis allows.
    # Test "all groups have the same dipole (up to the tiny kinematic differences)" directly, with several
    # ecliptic corrections (Step 6) and with a worst case in which each group has its own, independent
    # clustering with the data-driven model's large-scale power.
    from scipy.stats import chi2
    from dipole.templates import fit_general_mult, template_maps
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    rec = StageRecord(STAGE, "robust")
    z = np.load(config.RESULTS / "s14_ecliptic" / "templates.npz")
    single, _ = template_maps(ecliptic_latitude_map(), z["w1cov_pix"])
    cnts = [np.load(OUT / f"counts_g{g}.npy") for g in range(G)]
    fits_ = {"line (A)": (lambda c: fit_multipoles_mult(c, mask, ae), 0),
             "line+cov": (lambda c: fit_general_mult(c, mask, [single["line"], single["cov"]]), 0),
             "cov": (lambda c: fit_general_mult(c, mask, [single["cov"]]), 0),
             "line + l=2,3 (C)": (lambda c: fit_multipoles_mult(c, mask, ae, Y), 1),
             "line+cov + l=2,3": (lambda c: fit_general_mult(c, mask, [single["line"], single["cov"]], Y), 1)}
    wob = read_stage("s13_real", "wobble_data")["numbers"]["variance_per_component_by_ingredient"]
    out = {}
    for m in ("lcdm", "data"):
        Mk = np.concatenate([np.load(OUT / f"mocks_kin_{m}_{c}.npy") for c in range(3)])
        out[m] = {}
        for lab, (f, k) in fits_.items():
            ds = [f(c)["d"] for c in cnts]
            Dv = np.r_[ds[1] - ds[0], ds[2] - ds[0]]
            md = np.concatenate([Mk[:, k, 1] - Mk[:, k, 0], Mk[:, k, 2] - Mk[:, k, 0]], axis=1)
            mu, Cs = md.mean(0), np.cov(md.T)
            clus = sum(v["A" if k == 0 else "C"] for kk, v in wob.items() if kk != "Poisson noise")
            I3 = np.eye(3) * clus
            Ce = np.block([[2 * I3, I3], [I3, 2 * I3]])
            r = Dv - mu
            c_sh = float(r @ np.linalg.solve(Cs, r)); c_in = float(r @ np.linalg.solve(Cs + Ce, r))
            grp = []
            for d in ds:
                l, b = hp.vec2ang(d / np.linalg.norm(d), lonlat=True)
                e = SkyCoord(l[0] * u.deg, b[0] * u.deg, frame="galactic").barycentricmeanecliptic
                grp.append(dict(D=float(np.linalg.norm(d)), l=float(l[0]), b=float(b[0]), ecliptic_lat=float(e.lat.deg),
                                angle_to_CMB=float(np.degrees(np.arccos(d @ dhat / np.linalg.norm(d))))))
            out[m][lab] = dict(groups=grp, chi2_shared=c_sh, p_shared=float(chi2.sf(c_sh, 6)),
                               chi2_independent_worstcase=c_in, p_independent_worstcase=float(chi2.sf(c_in, 6)))
    rec.number("equal_dipole_tests", out)
    rec.note("Worst case adds, to each group, the full clustering variance of the data-driven model (Step 5 'wobble', "
             "all ingredients except Poisson noise) as if each group's clustering were independent of the others'.")
    rec.write()

elif part == "composition":
    # Added after "robust" (Nov 19-20). The Step 8 diagnosis (does the colour mix change across the sky?) was first
    # computed in throwaway commands; this part recomputes it so the numbers can be regenerated.
    #   (1) each group's share of the counts, in five bins of WISE coverage (w1cov), and the slope of the share
    #       against log10(w1cov) in units of its binomial standard error;
    #   (2) the dipole of each group's share map: share_p = f0 (1 + d . n_p), weighted least squares with
    #       binomial weights n_tot / (f0 (1 - f0));
    #   (3) the difference of the high- and low-alpha group dipoles (config A), for comparison with (2).
    # Binomial errors ignore any clustering of the colour mix, so (1) and (2) are indicative, not exact.
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    rec = StageRecord(STAGE, "composition")
    z = np.load(config.RESULTS / "s14_ecliptic" / "templates.npz")
    cov = z["w1cov_pix"]
    cnts = [np.load(OUT / f"counts_g{g}.npy").astype(float) for g in range(G)]
    tot = sum(cnts)
    use = seen & (tot > 0) & (cov > 0)
    V = pixel_vectors()[:, use]

    def direction(d):
        l, b = hp.vec2ang(d / np.linalg.norm(d), lonlat=True)
        e = SkyCoord(l[0] * u.deg, b[0] * u.deg, frame="galactic").barycentricmeanecliptic
        return dict(amp=float(np.linalg.norm(d)), l=float(l[0]), b=float(b[0]),
                    ecliptic_lon=float(e.lon.deg), ecliptic_lat=float(e.lat.deg),
                    angle_to_CMB=float(np.degrees(np.arccos(d @ dhat / np.linalg.norm(d)))))

    edges = np.quantile(cov[use], np.linspace(0, 1, 6))
    b_of = np.clip(np.searchsorted(edges, cov[use], side="right") - 1, 0, 4)
    out = {"w1cov_bin_edges": edges.tolist(), "groups": []}
    for g in range(G):
        n_g, n_t = cnts[g][use], tot[use]
        f0 = n_g.sum() / n_t.sum()
        shares = [float(n_g[b_of == k].sum() / n_t[b_of == k].sum()) for k in range(5)]
        # (1) weighted line share = c0 + c1 log10(w1cov) per pixel, binomial weights
        w = n_t / (f0 * (1 - f0))
        X = np.vstack([np.ones(use.sum()), np.log10(cov[use])])
        F = (X * w) @ X.T
        c = np.linalg.solve(F, (X * w) @ (n_g / n_t))
        z_slope = float(c[1] / np.sqrt(np.linalg.inv(F)[1, 1]))
        # (2) dipole of the share map
        X2 = np.vstack([np.ones(use.sum()), V])
        F2 = (X2 * w) @ X2.T
        c2 = np.linalg.solve(F2, (X2 * w) @ (n_g / n_t))
        dg = c2[1:] / c2[0]
        se = np.sqrt(np.diag(np.linalg.inv(F2))[1:]) / c2[0]
        out["groups"].append(dict(mean_share=float(f0), share_by_w1cov_bin=shares, slope_vs_log_w1cov=float(c[1]),
                                  slope_z=z_slope, share_dipole=direction(dg),
                                  share_dipole_amp_over_se=float(np.linalg.norm(dg) / np.sqrt(np.mean(se ** 2)))))
    ds = [fit_multipoles_mult(cc, mask, ae)["d"] for cc in cnts]
    out["high_minus_low_dipole_A"] = direction(ds[2] - ds[0])
    out["max_abs_slope_z"] = float(max(abs(gr["slope_z"]) for gr in out["groups"]))
    out["min_abs_slope_z"] = float(min(abs(gr["slope_z"]) for gr in out["groups"]))
    rec.number("composition", out)
    rec.write()

elif part == "summary":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rec = StageRecord(STAGE, "summary")
    sp = read_stage(STAGE, "split")["numbers"]["split"]["groups"]
    real = read_stage(STAGE, "real")["numbers"]["group_dipoles"]
    M = {(h, m): np.concatenate([np.load(OUT / f"mocks_{h}_{m}_{c}.npy") for c in range(3)])
         for h in ("kin", "fast", "intr") for m in ("lcdm", "data")}
    out, ok3, ok4 = {}, True, True
    for m in ("lcdm", "data"):
        out[m] = {}
        for k, cfg in enumerate(("A", "C")):
            Sig = np.cov(M[("kin", m)][:, k].reshape(len(M[("kin", m)]), -1).T)
            o = {}
            for R in RS:
                K = np.array([g[f"R{R}"]["K_direct"] for g in sp])
                t = tests(np.array(real[cfg]).ravel(), K, Sig)
                o[f"R{R}"] = dict(b=t["b"].tolist(), v_kms=t["v_kms"], b_sigma=np.sqrt(np.diag(t["Cb"])).tolist(),
                                  v_kms_sigma=float(np.sqrt((t["b"] / np.linalg.norm(t["b"])) @ t["Cb"] @ (t["b"] / np.linalg.norm(t["b"]))) * config.C_KMS),
                                  a=t["a"].tolist(), a_amp=float(np.linalg.norm(t["a"])), p_a0=t["p_a0"], p_bcmb=t["p_bcmb"],
                                  angle_b_to_CMB=float(np.degrees(np.arccos(t["b"] @ dhat / np.linalg.norm(t["b"])))),
                                  group_K=K.tolist(),
                                  group_D=[float(np.linalg.norm(x)) for x in real[cfg]],
                                  group_v_naive=[float(np.linalg.norm(x) / kk / beta * config.V_CMB_KMS) for x, kk in zip(real[cfg], K)])
            # calibration and power, R = 1
            K1 = np.array([g["R1.0"]["K_direct"] for g in sp])
            cal = {}
            for h in ("kin", "fast", "intr"):
                T = [tests(x.ravel(), K1, Sig) for x in M[(h, m)][:, k]]
                cal[h] = dict(frac_reject_a0=float(np.mean([t["p_a0"] < 0.05 for t in T])),
                              frac_reject_bcmb=float(np.mean([t["p_bcmb"] < 0.05 for t in T])),
                              median_v_kms=float(np.median([t["v_kms"] for t in T])),
                              mean_b=np.mean([t["b"] for t in T], axis=0).tolist(),
                              mean_a=np.mean([t["a"] for t in T], axis=0).tolist(),
                              se_b=(np.std([t["b"] for t in T], axis=0, ddof=1) / np.sqrt(len(T))).tolist())
                if h == "kin" and cfg == "A":
                    zb = (np.array(cal[h]["mean_b"]) - beta * dhat) / np.array(cal[h]["se_b"])
                    se_a = np.std([t["a"] for t in T], axis=0, ddof=1) / np.sqrt(len(T))
                    za = np.array(cal[h]["mean_a"]) / se_a
                    ok3 &= bool(np.all(np.abs(zb) < 3) and np.all(np.abs(za) < 3))
                    ok4 &= abs(cal[h]["frac_reject_a0"] - 0.05) < 0.03
            o["calibration_R1"] = cal
            out[m][cfg] = o
    rec.check("3. kinematic-only mocks: b unbiased (= beta_CMB toward the CMB) and a = 0 (config A, both models)", ok3)
    rec.check("4. kinematic-only mocks: 'a = 0 rejected' fires in 5% +/- 3% (config A, both models)", ok4,
              "; ".join(f"{m}: {out[m]['A']['calibration_R1']['kin']['frac_reject_a0']:.3f}" for m in out))
    rec.number("results", out)
    # Quaia implied velocity at its clean cuts (Step 7 'cuts'): v = v_CMB D / D_kin, error from the mocks' spread
    qv = {}
    for s, cuts in (("low", ("40", "45", "50")), ("high", ("45", "50"))):
        cu = read_stage("s15_quaia", f"cuts_{s}")["numbers"]["by_cut"]
        Dk = read_stage("s15_quaia", "dkin")["numbers"]["dkin"][s]["D_formula_onesided_bprp"]
        qv[s] = {c: dict(v_kms=cu[c]["A"]["D"] / Dk * config.V_CMB_KMS,
                         v_sigma_kms=cu[c]["A"]["sigma_D"] / Dk * config.V_CMB_KMS) for c in cuts}
    rec.number("quaia_velocity_naive", qv)
    rec.note("Quaia velocities are naive (D/D_kin x v_CMB): at these noise levels the measured amplitude is biased "
             "upward by the noise (Step 7 'cuts': the median amplitude of kinematic-only mocks exceeds D_kin by "
             "30-75%), so they are upper-biased and only indicative.")
    # figure: group dipole amplitude vs K, with the two hypotheses
    m, cfg = "lcdm", "A"
    o = out[m][cfg]["R1.0"]; K = np.array(o["group_K"])
    Dg = np.array(o["group_D"])
    Sig = np.cov(M[("kin", m)][:, 0].reshape(len(M[("kin", m)]), -1).T)
    eg = [np.sqrt(np.trace(Sig[3 * g:3 * g + 3, 3 * g:3 * g + 3]) / 3) for g in range(G)]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.errorbar(K, Dg * 1e3, yerr=np.array(eg) * 1e3, fmt="o", color="#0b0b0b", capsize=4,
                label="CatWISE groups (config A; error: 1σ per component)")
    for kk_, dd_, nm_, x_ in zip(K, Dg, ("low α", "middle α", "high α"), real[cfg]):
        ang_ = float(np.degrees(np.arccos(np.array(x_) @ dhat / np.linalg.norm(x_))))
        ax.annotate(f"{nm_}\n{ang_:.0f}° from CMB", (kk_, dd_ * 1e3), xytext=(10, -4), textcoords="offset points", fontsize=8)
    kk = np.linspace(K.min() - 0.5, K.max() + 0.5, 50)
    full = read_stage("s13_real", "dipole")["numbers"]["configs"]["A"]["D"]
    Dkf = read_stage("s13_real", "dkin")["numbers"]["dkin"]["D_direct"]
    ax.plot(kk, kk * beta * 1e3, ":", color="#7a7975", label="motion at the CMB speed (D = K β)")
    ax.plot(kk, kk * beta * full / Dkf * 1e3, "-", color="#2a78d6", label=f"faster motion ({full / Dkf:.2f} × CMB speed)")
    ax.plot(kk, (kk * beta + (full - Dkf)) * 1e3, "--", color="#eb6834", label="CMB motion + common excess")
    ax.set_xlabel("group's Doppler factor K = 2 + x(1 + α)"); ax.set_ylabel("dipole amplitude (× 10⁻³)")
    ax.legend(frameon=False, fontsize=9, loc="upper right"); ax.grid(alpha=0.3); ax.set_ylim(5, 33)
    ax.set_title("CatWISE split by spectral index: does the dipole scale with K?", loc="left", fontsize=11)
    plt.tight_layout(); plt.savefig(OUT / "origin_test.png", dpi=130); plt.close()
    rec.write()
else:
    sys.exit(f"unknown part {part}")
