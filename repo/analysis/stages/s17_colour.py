"""
Stage s17: does the full-sample dipole depend on how colour is measured and
selected? Follow-up to Step 8.

Step 8 found that the share of each colour group varies across the sky (with
scan depth and other patterns), so splitting by measured colour cannot test
the dipole's origin. Whatever moves measured colours around the sky can also
move quasars across the sample's own colour cut (W1 - W2 >= 0.8), which would
change total counts. Two tests:

A. Colour-cut scan: the final sample cut at W1 - W2 >= 0.80, 0.85, 0.90, 1.00,
   1.10 (nested samples). If colour-measurement effects at the cut drive the
   dipole, it should change as the cut moves away from where they act. Null
   mocks thin one clustered mock sky to each cut's size (nested, as the real
   samples are), with each cut's own kinematic dipole, so the real changes are
   compared with chance.
B. Colour template: a smoothed (10 deg) map of the low-alpha fraction (Step 8's
   group 0) added as a multiplicative correction in the full-sample fit; also
   the smoothed mean colour as an alternative. With Poisson counts the group
   fractions are independent of the total counts, so the template's noise does
   not correlate with the count noise it is fitted to.

Parts:
  maps          samples per cut, D_kin per cut (direct boost), colour templates
  real          dipoles per cut (configs A, A+cov, C) and with colour templates
  safety        noiseless planted skies: does a colour template absorb a real
                kinematic dipole? Does it remove a planted colour pattern?
  mocks <m> <c> nested null mocks (clustering m = lcdm, data; chunks c = 0..4
                of 60), every cut and fit as in 'real'
  summary       checks, results, figure

Checks (set before running)
  1. The samples are nested (counts per pixel never increase with the cut) and
     the 0.80 cut equals Step 5's final sample.
  2. Safety: on a noiseless kinematic sky, adding a colour template changes the
     recovered dipole by less than 2% of D_kin (both templates, configs A, C).
  3. Nested null mocks: the mean change of the dipole between cuts equals the
     expected change from the cuts' different D_kin (within 3 standard errors).
  4. Null mocks: the fitted colour-template amplitude has mean zero (within
     3 standard errors), so its spread calibrates the real amplitude.
"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, healpy as hp
from astropy.io import fits
import config
from dipole import load_mask, unit_vector, cos_theta_map, ecliptic_latitude_map, pixel_vectors, NPIX
from dipole.catalog import galactic_pixels, count_map
from dipole.measure import ecliptic_trend_fit
from dipole.mocks import expected_counts
from dipole.fluxlimit import millimag, d_direct_extrapolated
from dipole.clustering import gaussian_cls_for_lognormal, lognormal_delta, ylm_templates
from dipole.templates import fit_general_mult, template_maps
from dipole.checks import StageRecord, read_stage

STAGE = "s17_colour"
part = sys.argv[1] if len(sys.argv) > 1 else "summary"
OUT = config.stage_dir(STAGE)
LMAX = 3 * config.NSIDE - 1
I = json.loads((config.RESULTS / "s02_inputs" / "inputs.json").read_text())
beta = I["beta"]
mask = load_mask("s22"); seen = ~mask
elat = ecliptic_latitude_map(); ae = np.abs(elat)
dhat = unit_vector()
Y, _ = ylm_templates((2, 3))
CUTS = [0.80, 0.85, 0.90, 1.00, 1.10]
SMOOTH_DEG = 10.0


def tmaps():
    z = np.load(config.RESULTS / "s14_ecliptic" / "templates.npz")
    single, _ = template_maps(elat, z["w1cov_pix"])
    c = np.load(OUT / "colour_templates.npz")
    return single, c["frac_low"], c["mean_colour"]


def fits_for(cnt, single, flow, mcol):
    """Dipole vectors for every fit used in this stage; also the colour-template amplitudes."""
    L, C_ = single["line"], single["cov"]
    spec = {"A": ([L], None), "A+cov": ([L, C_], None), "C": ([L], Y),
            "A+frac": ([L, flow], None), "A+cov+frac": ([L, C_, flow], None), "C+frac": ([L, flow], Y),
            "A+colour": ([L, mcol], None)}
    out, amp = {}, {}
    for k, (tr, ex) in spec.items():
        f = fit_general_mult(cnt, mask, tr, ex)
        out[k] = f["d"]
        if k in ("A+frac", "A+cov+frac", "C+frac", "A+colour"):
            amp[k] = float(f["s"][-1] * np.std(tr[-1][seen]))      # fractional density change per 1 sd of template
    return out, amp


FITS = ["A", "A+cov", "C", "A+frac", "A+cov+frac", "C+frac", "A+colour"]

# ================================================================= parts
if part == "maps":
    rec = StageRecord(STAGE, "maps")
    c = fits.open(config.CATALOG)[1].data
    rows = np.load(config.RESULTS / "s13_real" / "final_rows.npy")
    w1, w12, al = np.array(c["w1"], float), np.array(c["w12"], float), np.array(c["alpha"], float)
    ipc = galactic_pixels(np.array(c["ra"]), np.array(c["dec"]))
    full = np.load(config.RESULTS / "s13_real" / "final_counts.npy")
    # fainter sources for the direct boost, as s13/s16
    sid = np.array(c["source_id"]).astype(str)
    faint = np.where((np.array(c["w1cov"]) >= config.W1COV_MIN) & (w1 >= config.W1_CUT) & seen[ipc])[0]
    nv = fits.open(config.RELEASE / "nvss" / "reference" / "NVSS.fit")[1].data
    nv_ids = np.array(nv["NVSS"]).astype(str)[np.load(config.RESULTS / "s13_real" / "nvss_sample_rows.npy")]
    mt = fits.open(config.RELEASE / "nvss" / "reference" / "NVSS_CatWISE2020_40arcsec_best_symmetric.fits")[1].data
    matched = np.array(mt["source_id"]).astype(str)[np.isin(np.array(mt["nvss"]).astype(str), nv_ids)]
    faint = faint[~np.isin(sid[faint], matched)]
    prev, nested, info = None, True, {}
    for cut in CUTS:
        r_ = rows[w12[rows] >= cut - 1e-9]; f_ = faint[w12[faint] >= cut - 1e-9]
        cnt = count_map(ipc[r_]).astype(float)
        np.save(OUT / f"counts_c{cut:.2f}.npy", cnt)
        if prev is not None:
            nested &= bool(np.all(cnt <= prev))
        prev = cnt
        p, e, _ = ecliptic_trend_fit(ae[seen], cnt[seen])
        MM = np.r_[millimag(w1[r_]), millimag(w1[f_])]; AL = np.r_[al[r_], al[f_]]
        info[f"{cut:.2f}"] = dict(n=int(r_.size), D_kin=d_direct_extrapolated(MM, AL, beta)[0],
                                  trend=dict(intercept=p[1], frac_slope=p[0] / p[1]))
    rec.check("1. samples are nested and the 0.80 cut is Step 5's final sample",
              nested and np.array_equal(np.load(OUT / "counts_c0.80.npy"), full),
              ", ".join(f"{k}: {v['n']:,}" for k, v in info.items()))
    rec.number("cuts", info)
    # colour templates: smoothed low-alpha fraction (Step 8 group 0) and smoothed mean W1-W2
    edges = read_stage("s16_origin", "split")["numbers"]["split"]["edges"]
    g0 = rows[al[rows] < edges[0]]
    n0 = count_map(ipc[g0]).astype(float)
    csum = np.bincount(ipc[rows], w12[rows], NPIX)
    sm = lambda m: hp.smoothing(np.where(seen, m, 0.0), fwhm=np.radians(SMOOTH_DEG), iter=1)
    ntot_s = sm(full)
    frac_low = np.where(seen, sm(n0) / np.where(ntot_s > 0, ntot_s, 1), 0.0)
    mean_col = np.where(seen, sm(csum) / np.where(ntot_s > 0, ntot_s, 1), 0.0)
    np.savez(OUT / "colour_templates.npz", frac_low=frac_low, mean_colour=mean_col)
    rec.number("frac_low_footprint_percentiles_5_50_95", np.percentile(frac_low[seen], [5, 50, 95]).tolist())
    rec.number("mean_colour_footprint_percentiles_5_50_95", np.percentile(mean_col[seen], [5, 50, 95]).tolist())
    single, _ = template_maps(elat, np.load(config.RESULTS / "s14_ecliptic" / "templates.npz")["w1cov_pix"])
    rec.number("corr_frac_low_with_cov_template", float(np.corrcoef(frac_low[seen], single["cov"][seen])[0, 1]))
    rec.number("corr_frac_low_with_abs_elat", float(np.corrcoef(frac_low[seen], ae[seen])[0, 1]))
    rec.write()

elif part == "real":
    rec = StageRecord(STAGE, "real")
    single, flow, mcol = tmaps()
    res, amps = {}, {}
    for cut in CUTS:
        d, a = fits_for(np.load(OUT / f"counts_c{cut:.2f}.npy"), single, flow, mcol)
        res[f"{cut:.2f}"] = {k: v.tolist() for k, v in d.items()}; amps[f"{cut:.2f}"] = a
    rec.number("dipoles", res); rec.number("colour_template_amplitudes", amps)
    rec.write()

elif part == "safety":
    rec = StageRecord(STAGE, "safety")
    single, flow, mcol = tmaps()
    info = read_stage(STAGE, "maps")["numbers"]["cuts"]["0.80"]
    Dk = info["D_kin"]
    base = expected_counts(cos_theta_map(dhat), ae, mask, N0=info["trend"]["intercept"], s=info["trend"]["frac_slope"], D=Dk)
    d, _ = fits_for(base, single, flow, mcol)
    err = {k: float(np.linalg.norm(v - Dk * dhat) / Dk) for k, v in d.items()}
    rec.number("kinematic_sky_error_over_Dkin", err)
    rec.check("2. on a noiseless kinematic sky, adding a colour template changes the dipole by < 2% of D_kin",
              all(err[k] < 0.02 for k in ("A+frac", "A+cov+frac", "C+frac", "A+colour")),
              ", ".join(f"{k}: {v:.4f}" for k, v in err.items()))
    # planted colour pattern at the amplitude fitted on the real sky (A+frac): what does it do without the template?
    amp = read_stage(STAGE, "real")["numbers"]["colour_template_amplitudes"]["0.80"]["A+frac"]
    t = flow - flow[seen].mean()
    planted = base * (1 + amp * t / t[seen].std())
    d2, _ = fits_for(planted, single, flow, mcol)
    rec.number("planted_colour_pattern", {k: dict(amp_over_Dkin=float(np.linalg.norm(v) / Dk),
                                                 angle=float(np.degrees(np.arccos(v @ dhat / np.linalg.norm(v)))))
                                         for k, v in d2.items()})
    rec.write()

elif part == "mocks":
    model, chunk = sys.argv[2], int(sys.argv[3])
    rec = StageRecord(STAGE, f"mocks_{model}_{chunk}")
    single, flow, mcol = tmaps()
    info = read_stage(STAGE, "maps")["numbers"]["cuts"]
    i0 = info["0.80"]
    cth = cos_theta_map(dhat)
    # full-sample expectation with the 0.80 kinematic dipole; thinning keeps each cut's share; each cut's
    # own kinematic dipole is restored by a per-pixel retention factor (1 + (Dk_c - Dk_0.8) cos theta)
    lam = expected_counts(cth, ae, mask, N0=i0["trend"]["intercept"], s=i0["trend"]["frac_slope"], D=i0["D_kin"])
    w = np.load(config.RESULTS / "s11_clustering" / "pixwin.npy")[: LMAX + 1]
    clg = gaussian_cls_for_lognormal(np.load(config.RESULTS / "s11_clustering" / f"cl_{model}.npy")[: LMAX + 1] * w ** 2, LMAX)
    pv = read_stage("s11_clustering", "fit")["numbers"]["subpixel_pixel_variance"]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 60
    D = np.zeros((n, len(CUTS), len(FITS), 3)); A = np.zeros((n, len(CUTS), 4))
    for j in range(n):
        i = chunk * n + j
        rng = np.random.default_rng(32_000_000 + i)
        mu = lam * (1 + lognormal_delta(clg, LMAX, rng)) * rng.gamma(1 / pv, pv, NPIX)
        cm = rng.poisson(mu)
        prev = cm
        for k, cut in enumerate(CUTS):
            if k > 0:
                ic = info[f"{cut:.2f}"]; ip_ = info[f"{CUTS[k - 1]:.2f}"]
                keep = ic["n"] / ip_["n"] * (1 + (ic["D_kin"] - ip_["D_kin"]) * cth)
                prev = rng.binomial(prev, np.clip(keep, 0, 1))
            d, a = fits_for(prev.astype(float), single, flow, mcol)
            D[j, k] = [d[f] for f in FITS]
            A[j, k] = [a[f] for f in ("A+frac", "A+cov+frac", "C+frac", "A+colour")]
    np.savez(OUT / f"mocks_{model}_{chunk}.npz", D=D, A=A)
    rec.number("n", n)
    rec.write()

elif part == "summary":
    from scipy.stats import chi2
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rec = StageRecord(STAGE, "summary")
    info = read_stage(STAGE, "maps")["numbers"]["cuts"]
    real = read_stage(STAGE, "real")["numbers"]
    M = {}
    for m in ("lcdm", "data"):
        z = [np.load(OUT / f"mocks_{m}_{c}.npz") for c in range(5)]
        M[m] = dict(D=np.concatenate([q["D"] for q in z]), A=np.concatenate([q["A"] for q in z]))
    ok3, ok4 = True, True
    res = {}
    for m in M:
        res[m] = {}
        for fi, fit in enumerate(FITS):
            rows = {}
            for k, cut in enumerate(CUTS):
                kc = f"{cut:.2f}"; Dk = info[kc]["D_kin"]
                d = np.array(real["dipoles"][kc][fit])
                x = M[m]["D"][:, k, fi]
                r = dict(D=float(np.linalg.norm(d)), D_over_Dkin=float(np.linalg.norm(d) / Dk),
                         angle_to_CMB=float(np.degrees(np.arccos(d @ dhat / np.linalg.norm(d)))),
                         sigma=float(np.sqrt(np.mean(np.var(x, axis=0, ddof=1)))))
                if k > 0:
                    d0 = np.array(real["dipoles"]["0.80"][fit]); dd = d - d0
                    md = x - M[m]["D"][:, 0, fi]
                    mu, Cm = md.mean(0), np.cov(md.T)
                    expected = (Dk - info["0.80"]["D_kin"]) * dhat
                    if fit == "A":
                        zz = (mu - expected) / (md.std(0, ddof=1) / np.sqrt(len(md)))
                        ok3 &= bool(np.all(np.abs(zz) < 3))
                    c2 = float((dd - mu) @ np.linalg.solve(Cm, dd - mu))
                    r.update(change_from_0p80=float(np.linalg.norm(dd)), chance_change_rms=float(np.sqrt(np.trace(Cm))),
                             p_change_from_0p80=float(chi2.sf(c2, 3)))
                rows[kc] = r
            res[m][fit] = rows
        amps = {}
        for ai, nm in enumerate(("A+frac", "A+cov+frac", "C+frac", "A+colour")):
            x = M[m]["A"][:, 0, ai]
            if abs(x.mean()) >= 3 * x.std(ddof=1) / np.sqrt(len(x)):
                ok4 = False
            ra = real["colour_template_amplitudes"]["0.80"][nm]
            amps[nm] = dict(real=ra, null_mean=float(x.mean()), null_sd=float(x.std(ddof=1)),
                            z=float((ra - x.mean()) / x.std(ddof=1)))
        res[m]["colour_template_amplitude_0p80"] = amps
    rec.check("3. nested null mocks: mean dipole change between cuts = expected from D_kin (config A)", ok3)
    rec.check("4. null mocks: colour-template amplitude has mean zero", ok4)
    rec.number("results", res)
    # direction of each cut's dipole (config A and A+cov+frac) in Galactic and ecliptic coordinates
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    dirs = {}
    for fit in ("A", "A+cov+frac"):
        dirs[fit] = {}
        for cut in CUTS:
            d = np.array(real["dipoles"][f"{cut:.2f}"][fit])
            l, b = hp.vec2ang(d / np.linalg.norm(d), lonlat=True)
            e = SkyCoord(l[0] * u.deg, b[0] * u.deg, frame="galactic").barycentricmeanecliptic
            dirs[fit][f"{cut:.2f}"] = dict(l=float(l[0]), b=float(b[0]), ecliptic_lon=float(e.lon.deg), ecliptic_lat=float(e.lat.deg))
    rec.number("directions", dirs)
    rec.note("Moving the colour cut also moves where colour noise acts: noise changes the counts above a cut in "
             "proportion to the slope of the colour distribution at the cut, which changes sign near its peak "
             "(W1 - W2 ~ 1.0-1.1). The per-cut ecliptic slope falls from -0.00093 to -0.00012 per deg between the "
             "0.80 and 1.10 cuts, consistent with this. The scan therefore shows that the dipole depends on the "
             "colour selection; it cannot by itself say whether a noise effect at each cut or a real difference "
             "between redder and bluer quasars is responsible.")
    # figure
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    xs = np.array(CUTS)
    for fit, col in (("A", "#0b0b0b"), ("A+cov", "#2a78d6"), ("C", "#eb6834")):
        ys = [res["data"][fit][f"{c:.2f}"]["D_over_Dkin"] for c in CUTS]
        es = [res["data"][fit][f"{c:.2f}"]["sigma"] / info[f"{c:.2f}"]["D_kin"] for c in CUTS]
        ax[0].errorbar(xs + {"A": -0.006, "A+cov": 0, "C": 0.006}[fit], ys, yerr=es, fmt="o-", color=col, capsize=3, label=fit)
    ax[0].axhline(1, color="k", ls=":", lw=1)
    ax[0].set_xlabel("colour cut: W1 − W2 ≥"); ax[0].set_ylabel("dipole ÷ that sample's D_kin\n(error bars: data-driven null spread)")
    ax[0].legend(frameon=False); ax[0].grid(alpha=0.3)
    ax[0].set_title("A. Does the dipole change as the colour cut moves?", loc="left", fontsize=11)
    labs = ["A", "A+frac", "A+cov", "A+cov+frac", "C", "C+frac", "A+colour"]
    vals = [res["data"][f]["0.80"]["D_over_Dkin"] for f in labs]
    errs = [res["data"][f]["0.80"]["sigma"] / info["0.80"]["D_kin"] for f in labs]
    ax[1].errorbar(range(len(labs)), vals, yerr=errs, fmt="o", color="#0b0b0b", capsize=3)
    ax[1].axhline(1, color="k", ls=":", lw=1)
    ax[1].set_xticks(range(len(labs))); ax[1].set_xticklabels(labs, rotation=25, ha="right")
    ax[1].set_ylabel("dipole ÷ D_kin (full sample, W1 − W2 ≥ 0.8)"); ax[1].grid(axis="y", alpha=0.3)
    ax[1].set_title("B. With and without a colour-composition correction", loc="left", fontsize=11)
    plt.tight_layout(); plt.savefig(OUT / "colour_tests.png", dpi=130); plt.close()
    rec.write()
else:
    sys.exit(f"unknown part {part}")
