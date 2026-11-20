"""
Stage s15: the Quaia dipole. Project Step 7.

Samples: "low" (G < 20.0, 755,850 quasars) and "high" (G < 20.5, 1,295,502),
each with its own selection function S (equatorial HEALPix, N_side = 64).
Galactic cuts |b| >= 0, 10, 20, 30, 40 deg, as Mittal et al. (2024); pixels
with S = 0 are always excluded.

Parts:
  maps            count maps, masks, templates on the equatorial grid
  mittal          Mittal et al.'s model (lam = a S (1 + d.n), Poisson maximum
                  likelihood) at every cut, both samples
  dkin            spectral indices (BP-RP; G-BP as a cross-check), x at the G
                  limit, D_kin (formula; direct boost for "low", which has the
                  G < 20.5 catalog's fainter sources available)
  configs         the dipole at every cut in three configurations:
                    A: S only (Mittal et al.'s model)
                    C: S and l = 2, 3 templates
                    A+scan: S times (1 + s x unWISE scan depth^-1/2)
  nz              redshift distribution of each sample (for the theory spectrum)
  theory          CAMB spectrum for each sample (run anywhere; uses only N(z))
  clfit <sample>  clustering models for the sample at the 30-deg cut, fitted to
                  its measured spectrum by forward modelling (as Step 4):
                  lcdm (one amplitude, l >= 26) and data-driven (per band)
  mocks <sample> <cut> <model> <chunk>   null mocks: kinematic dipole x S x
                  clustering x sub-pixel scatter, Poisson; configs A and C
  summary         checks, results, significance, figure
  cuts <sample> <n>  (added later) cuts 30-60 deg: is each further cut's change in
                  the dipole larger than chance (null mocks fitted at every cut on
                  the same sky)? Is there leftover l = 2, 3, 5 power above lcdm?
                  How compatible is the measured amplitude with 1 x and 2 x D_kin?

Checks (set before running)
  1. The selection function correlates with the counts on the equatorial grid
     (r > 0.5) and not on the Galactic grid (|r| < 0.2), confirming its frame.
  2. Our Mittal-model amplitudes reproduce Mittal et al.'s (Fig. 6, Poisson
     likelihood, free dipole) within 3 x 10^-3 at every cut, both samples.
  3. Poisson maximum likelihood and least squares agree within 0.3 standard
     deviations (Mittal model, 30-deg cut, both samples).
  4. D_kin for "low": the formula with local x, alpha equals the direct boost
     within 0.5%.
  5. Null mocks, config A: unbiased (mean vector = D_kin toward the CMB within
     3 standard errors) for every sample, cut and model.
"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, healpy as hp
from astropy.io import fits
import config
from dipole import NPIX
from dipole.quaia import (eq_pixel_vectors, galactic_latitude_of_eq_pixels, eq_vector, to_galactic, spectral_index,
                          fit_offset_lsq, fit_offset_ml)
from dipole.fluxlimit import millimag, x_local, alpha_local, d_direct_extrapolated
from dipole.measure import d_kin
from dipole.clustering import gaussian_cls_for_lognormal, lognormal_delta, ylm_templates, band, BANDS
from dipole.checks import StageRecord, read_stage

STAGE = "s15_quaia"
part = sys.argv[1] if len(sys.argv) > 1 else "summary"
OUT = config.stage_dir(STAGE)
Q = config.PROJECT / "quaia"
SAMPLES = {"low": 20.0, "high": 20.5}
CUTS = [0, 10, 20, 30, 40]
MITTAL = {"low": {0: 37, 10: 34, 20: 25, 30: 17, 40: 11}, "high": {0: 48, 10: 43, 20: 32, 30: 20, 40: 12}}  # x 1e-3, Fig. 6
LMAX = 3 * config.NSIDE - 1
I = json.loads((config.RESULTS / "s02_inputs" / "inputs.json").read_text())
beta = I["beta"]
cmb_eq = eq_vector(config.L_CMB_DEG, config.B_CMB_DEG)
Y, _ = ylm_templates((2, 3))
HIGHB = [i for i, (a, b) in enumerate(BANDS) if a >= 26]


def load(sample):
    return fits.open(Q / f"quaia_G{SAMPLES[sample]:.1f}.fits")[1].data


def sel(sample):
    return hp.read_map(str(Q / f"selection_function_NSIDE64_G{SAMPLES[sample]:.1f}.fits"))


def masks(S):
    bg = galactic_latitude_of_eq_pixels()
    return {c: (np.abs(bg) < c) | (S <= 0) for c in CUTS}


def counts(sample):
    return np.load(OUT / f"counts_{sample}.npy")


def scan_template():
    m = np.load(Q / "selection_function_template_maps" / "selection_function_template_maps" / "map_unwisescan_NSIDE64.npy")
    return m ** -0.5


def angle(a, b):
    return float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))


def residual_bands(cnt, mask, S):
    f = fit_offset_lsq(cnt, mask, S)
    seen = ~mask
    lam = f["a"] * S * (1 + eq_pixel_vectors().T @ f["d"])
    r = np.zeros(NPIX); r[seen] = cnt[seen] / lam[seen] - 1
    return band(hp.anafast(r, lmax=LMAX) / seen.mean())


def clustered_mean(lam, clg, pixvar, rng):
    mu = lam * (1 + lognormal_delta(clg, LMAX, rng))
    return mu * rng.gamma(1 / pixvar, pixvar, NPIX) if pixvar > 0 else mu


# ================================================================= parts
if part == "maps":
    rec = StageRecord(STAGE, "maps")
    for s in SAMPLES:
        d = load(s)
        ip = hp.ang2pix(config.NSIDE, np.array(d["ra"]), np.array(d["dec"]), lonlat=True)
        cnt = np.bincount(ip, minlength=NPIX).astype(float)
        np.save(OUT / f"counts_{s}.npy", cnt)
        S = sel(s)
        ipg = hp.ang2pix(config.NSIDE, np.array(d["l"]), np.array(d["b"]), lonlat=True)
        cg = np.bincount(ipg, minlength=NPIX)
        k = S > 0
        r_eq, r_g = float(np.corrcoef(cnt[k], S[k])[0, 1]), float(np.corrcoef(cg[k], S[k])[0, 1])
        rec.number(f"corr_counts_S_{s}", dict(equatorial=r_eq, galactic=r_g))
        rec.check(f"1. selection function is equatorial ({s})", r_eq > 0.5 and abs(r_g) < 0.2, f"r = {r_eq:.3f} vs {r_g:.3f}")
        rec.number(f"n_{s}", int(len(d)))
        M = masks(S)
        rec.number(f"n_in_cut_{s}", {c: int(cnt[~M[c]].sum()) for c in CUTS})
        rec.number(f"fsky_{s}", {c: float((~M[c]).mean()) for c in CUTS})
    rec.write()

elif part == "mittal":
    rec = StageRecord(STAGE, "mittal")
    out, ok, ok3 = {}, True, True
    for s in SAMPLES:
        cnt, S = counts(s), sel(s); M = masks(S); out[s] = {}
        for c in CUTS:
            f = fit_offset_ml(cnt, M[c], S)
            l, b = to_galactic(f["d"])
            out[s][c] = dict(D=f["amp"], l=l, b=b, angle_to_CMB=angle(f["d"], cmb_eq), converged=f["converged"],
                             D_lsq=f["lsq"]["amp"], mittal=MITTAL[s][c] * 1e-3,
                             z_ml_vs_lsq=((f["d"] - f["lsq"]["d"]) / f["lsq"]["sigma"]).tolist())
            ok &= abs(f["amp"] - MITTAL[s][c] * 1e-3) < 3e-3
            if c == 30:
                ok3 &= bool(np.all(np.abs(out[s][c]["z_ml_vs_lsq"]) < 0.3))
    rec.number("mittal_model", out)
    # Variants, added after check 2 failed: which weighting reproduces Mittal et al.'s Fig. 6?
    from dipole import fit_dipole_lsq
    var = {}
    for s in SAMPLES:
        cnt, S = counts(s), sel(s); M = masks(S)
        scaled = np.where(S > 0, cnt / np.where(S > 0, S, 1), 0)
        var[s] = {"poisson_ml_S_offset": [out[s][c]["D"] for c in CUTS],
                  "counts_over_S_unweighted": [fit_dipole_lsq(scaled, M[c])["amp"] for c in CUTS],
                  "no_selection_function": [fit_offset_ml(cnt, M[c], np.where(S > 0, 1.0, 0.0))["amp"] for c in CUTS],
                  "S_offset_pixels_S_gt_0.5": [fit_offset_ml(cnt, M[c] | (S < 0.5), S)["amp"] for c in CUTS]}
    rec.number("variants_by_cut", var)
    rec.note("Check 2 fails at small Galactic cuts: our Poisson fit with S as a fixed factor gives lower amplitudes "
             "than Mittal et al. (low: 29, 27, 21, 15, 8 vs 37, 34, 25, 17, 11 x 1e-3). Dividing the counts by S and "
             "fitting with equal weights per pixel reproduces them much better at |b| >= 10 deg (low: 36, 26, 16, 8; "
             "high: 44, 31, 21, 11 vs 43, 32, 20, 12). That weighting gives low-S pixels near the Galactic plane far "
             "more influence. Mittal et al. describe a Poisson likelihood with S multiplying the rate, so the cause is "
             "not certain. All variants agree within ~2e-3 at the 30 and 40 deg cuts, which the analysis uses.")
    rec.check("2. Mittal-model amplitudes reproduce Mittal et al.'s Fig. 6 within 3e-3 (all cuts, both samples)", ok,
              "; ".join(f"{s}: " + ", ".join(f"{c}: {out[s][c]['D']*1e3:.1f} ({MITTAL[s][c]})" for c in CUTS) for s in SAMPLES))
    rec.check("3. Poisson ML = least squares within 0.3 sd (30-deg cut)", ok3,
              "; ".join(f"{s}: {np.round(out[s][30]['z_ml_vs_lsq'], 2).tolist()}" for s in SAMPLES))
    rec.write()

elif part == "dkin":
    rec = StageRecord(STAGE, "dkin")
    lo, hi = load("low"), load("high")
    out = {}
    for s, d in (("low", lo), ("high", hi)):
        g, bp, rp = (np.array(d[k], float) for k in ("phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag"))
        a_bprp = spectral_index(bp, rp, "BP", "RP")
        a_gbp = spectral_index(g, bp, "G", "BP")
        lim = int(round(SAMPLES[s] * 1000))
        mm = millimag(g)
        r = dict(alpha_bprp_mean=float(np.mean(a_bprp)), alpha_gbp_mean=float(np.mean(a_gbp)))
        r["x_local_onesided"] = x_local(mm, lim, half=100, deg=2, one_sided=True)
        r["alpha_bprp_at_limit"] = alpha_local(mm, a_bprp, lim, half=100, one_sided=True)
        r["alpha_gbp_at_limit"] = alpha_local(mm, a_gbp, lim, half=100, one_sided=True)
        r["D_formula_onesided_bprp"] = d_kin(r["x_local_onesided"], r["alpha_bprp_at_limit"], beta)
        r["D_formula_onesided_gbp"] = d_kin(r["x_local_onesided"], r["alpha_gbp_at_limit"], beta)
        out[s] = r
    # "low": sources just fainter than G = 20.0 exist in the G < 20.5 catalog
    g, bp, rp = (np.array(hi[k], float) for k in ("phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag"))
    mm, a = millimag(g), spectral_index(bp, rp, "BP", "RP")
    out["low"]["x_local_twosided"] = x_local(mm, 20000, half=100, deg=2)
    out["low"]["alpha_bprp_twosided"] = alpha_local(mm, a, 20000, half=100)
    out["low"]["D_formula_twosided"] = d_kin(out["low"]["x_local_twosided"], out["low"]["alpha_bprp_twosided"], beta)
    out["low"]["D_direct"] = d_direct_extrapolated(mm, a, beta, m_lim=20.0)[0]
    out["note"] = ("The G < 20.0 catalog is a subset of the G < 20.5 catalog, but Quaia's classifier also depends on "
                   "other quantities; the direct boost for 'low' assumes the G < 20.5 sources just fainter than 20.0 "
                   "are the ones that would cross the cut.")
    rec.number("dkin", out)
    rec.number("mittal_erratum", dict(low=0.0048, high=0.0043))
    e = abs(out["low"]["D_formula_twosided"] / out["low"]["D_direct"] - 1)
    rec.check("4. D_kin (low): formula with local x, alpha = direct boost within 0.5%", e < 5e-3,
              f"formula {out['low']['D_formula_twosided']:.5f}, direct {out['low']['D_direct']:.5f}")
    rec.write()

elif part == "configs":
    rec = StageRecord(STAGE, "configs")
    tscan = scan_template()
    out = {}
    for s in SAMPLES:
        cnt, S = counts(s), sel(s); M = masks(S); out[s] = {}
        for c in CUTS:
            r = {}
            for nm, kw in (("A", {}), ("C", dict(extra=Y)), ("A+scan", dict(trends=[tscan]))):
                # least squares: the Poisson ML optimizer diverged for C and A+scan in a first run (it can step
                # into parameters where a pixel's predicted count is negative); LSQ = ML for config A (check 3)
                f = fit_offset_lsq(cnt, M[c], S, **kw)
                l, b = to_galactic(f["d"])
                r[nm] = dict(D=f["amp"], l=l, b=b, angle_to_CMB=angle(f["d"], cmb_eq), d_eq=f["d"].tolist(),
                             s=f["s"].tolist(), sigma_formal=f["sigma"].tolist())
            out[s][c] = r
    rec.number("configs", out)
    rec.write()

elif part == "nz":
    rec = StageRecord(STAGE, "nz")
    zg = np.linspace(0, 4.6, 231)
    for s in SAMPLES:
        z = np.array(load(s)["redshift_quaia"], float)
        h, e = np.histogram(z, bins=zg, density=True)
        rec.number(f"nz_{s}", dict(z=(0.5 * (e[1:] + e[:-1])).tolist(), pdf=h.tolist(), mean=float(z.mean())))
    rec.write()

elif part == "theory":
    from dipole.clustering import PLANCK18, bias_tiwari, exact_linear_cls
    import camb
    from camb.sources import SplinedSourceWindow
    rec = StageRecord(STAGE, "theory")
    nz = read_stage(STAGE, "nz")["numbers"]
    for s in SAMPLES:
        z = np.r_[0.0, nz[f"nz_{s}"]["z"]]; p = np.r_[0.0, nz[f"nz_{s}"]["pdf"]]
        from scipy.ndimage import gaussian_filter1d
        p = gaussian_filter1d(p, 2)                              # smooth the histogram (bin 0.02) for the spline
        zg = np.linspace(0, 4.6, 921); n = np.clip(np.interp(zg, z, p), 0, None); n /= np.trapezoid(n, zg)
        bz = bias_tiwari(zg)
        pars = camb.set_params(**PLANCK18, lmax=260)
        pars.Want_CMB = False; pars.NonLinear = camb.model.NonLinear_both
        st = pars.SourceTerms
        for k in ("counts_redshift", "counts_lensing", "counts_velocity", "counts_radial", "counts_timedelay",
                  "counts_ISW", "counts_potential"):
            setattr(st, k, k in ("counts_redshift", "counts_lensing"))
        st.counts_evolve = False
        # magnification: dlog10N/dm = 0.4 x, x from part dkin (one-sided local)
        x = read_stage(STAGE, "dkin")["numbers"]["dkin"][s]["x_local_onesided"]
        pars.SourceWindows = [SplinedSourceWindow(z=zg[1:], W=n[1:], bias_z=bz[1:], dlog10Ndm=0.4 * x)]
        pars.set_accuracy(AccuracyBoost=3, lSampleBoost=1, lAccuracyBoost=1)
        res = camb.get_results(pars)
        cl = res.get_source_cls_dict(raw_cl=True)["W1xW1"][:261].copy(); cl[0] = 0
        cl[1] = exact_linear_cls(res, zg, n, bz, ells=(1,), nk=6000, kmax=0.5)[1]
        np.save(OUT / f"theory_cls_{s}.npy", cl)
        rec.number(f"C1_{s}", cl[1])
    rec.note("Same method as s11 (Tiwari et al.'s bias shape; amplitude fitted to the data). N(z) from Quaia's own "
             "redshifts (redshift_quaia), smoothed.")
    rec.write()

elif part == "clfit":
    s = sys.argv[2]
    rec = StageRecord(STAGE, f"clfit_{s}")
    cnt, S = counts(s), sel(s); mask = masks(S)[30]
    w = np.load(config.RESULTS / "s11_clustering" / "pixwin.npy")[: LMAX + 1]
    cl_th = np.load(OUT / f"theory_cls_{s}.npy")[: LMAX + 1]
    f0 = fit_offset_lsq(cnt, mask, S)
    lam = f0["a"] * S * (1 + 0.0 * eq_pixel_vectors()[0]); lam[mask] = 0
    Pd = residual_bands(cnt, mask, S)
    gen = lambda cl: gaussian_cls_for_lognormal(cl * w ** 2, LMAX)
    pv_unit = read_stage("s11_clustering", "fit")["numbers"]["subpixel_pixel_variance"] / \
        read_stage("s11_clustering", "fit")["numbers"]["lcdm_amplitude"]

    def mock_bands(clg, pixvar, seeds):
        out = []
        for sd in seeds:
            rng = np.random.default_rng(sd)
            mu = lam if clg is None else clustered_mean(lam, clg, pixvar, rng)
            out.append(residual_bands(rng.poisson(mu).astype(float), mask, S))
        return np.array(out)
    N = mock_bands(None, 0, [26_000_000 + i for i in range(100)]).mean(0)
    tb = band(cl_th * w ** 2)
    A = 1.0
    for it in range(4):
        M_ = mock_bands(gen(A * cl_th), A * pv_unit, [26_100_000 + 1000 * it + i for i in range(150)]).mean(0)
        A *= np.sum(Pd[HIGHB] - N[HIGHB]) / np.sum(M_[HIGHB] - N[HIGHB])
    r = np.ones(len(BANDS))
    for it in range(8):
        cl_d = A * cl_th.copy()
        for j, (a, b) in enumerate(BANDS):
            cl_d[a:b + 1] *= r[j]
        M_ = mock_bands(gen(cl_d), A * pv_unit, [26_200_000 + 1000 * it + i for i in range(200)]).mean(0)
        r = np.maximum(0.0, r + (Pd - M_) / (A * tb))
    cl_data = A * cl_th.copy()
    for j, (a, b) in enumerate(BANDS):
        cl_data[a:b + 1] *= r[j]
    np.save(OUT / f"cl_lcdm_{s}.npy", A * cl_th); np.save(OUT / f"cl_data_{s}.npy", cl_data)
    rec.number("lcdm_amplitude", A); rec.number("data_band_factors", r.tolist())
    rec.number("subpixel_pixel_variance", A * pv_unit)
    rec.note("The sub-pixel scatter is scaled from s11 (CatWISE) by the ratio of fitted amplitudes, an approximation: "
             "Quaia's spectrum beyond l = 191 was not computed separately.")
    rec.write()

elif part == "mocks":
    s, cut, model, chunk = sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
    rec = StageRecord(STAGE, f"mocks_{s}_{cut}_{model}_{chunk}")
    cnt, S = counts(s), sel(s); mask = masks(S)[cut]
    Dk = read_stage(STAGE, "dkin")["numbers"]["dkin"][s]["D_formula_onesided_bprp"]
    f0 = fit_offset_lsq(cnt, mask, S)
    lam = f0["a"] * S * (1 + Dk * (eq_pixel_vectors().T @ cmb_eq)); lam[mask] = 0
    w = np.load(config.RESULTS / "s11_clustering" / "pixwin.npy")[: LMAX + 1]
    clg = gaussian_cls_for_lognormal(np.load(OUT / f"cl_{model}_{s}.npy")[: LMAX + 1] * w ** 2, LMAX)
    pv = read_stage(STAGE, f"clfit_{s}")["numbers"]["subpixel_pixel_variance"]
    out = np.zeros((125, 2, 3))
    for j in range(125):
        i = chunk * 125 + j
        rng = np.random.default_rng(27_000_000 + 1000 * cut + i)
        cm = rng.poisson(clustered_mean(lam, clg, pv, rng)).astype(float)
        out[j] = [fit_offset_lsq(cm, mask, S)["d"], fit_offset_lsq(cm, mask, S, extra=Y)["d"]]
    np.save(OUT / f"mocks_{s}_{cut}_{model}_{chunk}.npy", out)
    rec.number("n", 125)
    rec.write()

elif part == "cuts":
    # Added after Step 7 (prompted by "how do we quantify that 40 degrees is the cleanest?").
    # For each cut from 30 to 60 deg: (1) stability - the change in the real dipole from the previous cut,
    # against the change that losing that sky produces by chance in null mocks (lcdm clustering, no
    # contamination), fitted at every cut on the SAME mock sky; (2) leftover large-scale power at l = 2, 3, 5
    # against the same mocks. Configs A and C. Sample given on the command line.
    s = sys.argv[2]
    rec = StageRecord(STAGE, f"cuts_{s}")
    ccuts = [30, 35, 40, 45, 50, 55, 60]
    cnt, S = counts(s), sel(s)
    bg = galactic_latitude_of_eq_pixels()
    MK = {c: (np.abs(bg) < c) | (S <= 0) for c in ccuts}
    Dk = read_stage(STAGE, "dkin")["numbers"]["dkin"][s]["D_formula_onesided_bprp"]
    w = np.load(config.RESULTS / "s11_clustering" / "pixwin.npy")[: LMAX + 1]
    clg = gaussian_cls_for_lognormal(np.load(OUT / f"cl_lcdm_{s}.npy")[: LMAX + 1] * w ** 2, LMAX)
    pv = read_stage(STAGE, f"clfit_{s}")["numbers"]["subpixel_pixel_variance"]
    base = MK[ccuts[0]]
    f0 = fit_offset_lsq(cnt, base, S)
    lam = f0["a"] * S * (1 + Dk * (eq_pixel_vectors().T @ cmb_eq)); lam[S <= 0] = 0
    sel_b = [0, 1, 3]                                     # BANDS: l = 2, 3, 5
    real = {c: dict(A=fit_offset_lsq(cnt, MK[c], S), C=fit_offset_lsq(cnt, MK[c], S, extra=Y)) for c in ccuts}
    real_bands = {c: residual_bands(cnt, MK[c], S)[sel_b] for c in ccuts}
    nm = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    MD = {k: np.zeros((nm, len(ccuts), 3)) for k in ("A", "C")}; MB = np.zeros((nm, len(ccuts), 3))
    for i in range(nm):
        rng = np.random.default_rng(28_000_000 + i)
        cm = rng.poisson(clustered_mean(lam, clg, pv, rng)).astype(float)
        for j, c in enumerate(ccuts):
            MD["A"][i, j] = fit_offset_lsq(cm, MK[c], S)["d"]
            MD["C"][i, j] = fit_offset_lsq(cm, MK[c], S, extra=Y)["d"]
            MB[i, j] = residual_bands(cm, MK[c], S)[sel_b]
    out = {}
    for j, c in enumerate(ccuts):
        r = {"fsky": float((~MK[c]).mean())}
        for k in ("A", "C"):
            d = real[c][k]["d"]
            r[k] = dict(D=float(np.linalg.norm(d)), D_over_Dkin=float(np.linalg.norm(d) / Dk), angle_to_CMB=angle(d, cmb_eq),
                        sigma_D=float(np.sqrt(np.mean(np.var(MD[k][:, j, :], axis=0, ddof=1)))))
            if j > 0:
                dd = d - real[ccuts[j - 1]][k]["d"]
                md = MD[k][:, j, :] - MD[k][:, j - 1, :]
                cov = np.cov(md.T)
                chi2v = float(dd @ np.linalg.solve(cov, dd))
                from scipy.stats import chi2 as _c2
                r[k]["change_from_previous_cut"] = float(np.linalg.norm(dd))
                r[k]["chance_change_rms"] = float(np.sqrt(np.trace(cov)))
                r[k]["p_change_by_chance"] = float(_c2.sf(chi2v, 3))
        # compatibility of the measured amplitude with 1 x and 2 x D_kin (config A): mock vectors (true dipole
        # D_kin) give the noise; the 2 x hypothesis adds D_kin toward the CMB to every mock vector.
        dA = real[c]["A"]["d"]; Dobs = float(np.linalg.norm(dA))
        x1 = MD["A"][:, j, :]; x2 = x1 + Dk * cmb_eq
        g_ = np.random.default_rng(29)
        def frac(x, ge):
            mu, Lc = x.mean(0), np.linalg.cholesky(np.cov(x.T))
            a_ = np.linalg.norm(mu + g_.standard_normal((2_000_000, 3)) @ Lc.T, axis=1)
            return float(np.mean(a_ >= Dobs) if ge else np.mean(a_ <= Dobs))
        r["A"]["p_as_high_if_1x"] = frac(x1, True)
        r["A"]["p_as_low_if_2x"] = frac(x2, False)
        r["A"]["mock_median_amp_1x"] = float(np.median(np.linalg.norm(x1, axis=1)))
        r["A"]["mock_median_amp_2x"] = float(np.median(np.linalg.norm(x2, axis=1)))
        mb, sb = MB[:, j, :].mean(0), MB[:, j, :].std(0, ddof=1)
        r["leftover_power_z_vs_lcdm"] = dict(zip(["l=2", "l=3", "l=5"], ((real_bands[c] - mb) / sb).tolist()))
        out[c] = r
    rec.number("by_cut", out)
    rec.number("n_mocks", nm)
    rec.write()

elif part == "summary":
    from scipy.stats import norm
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rec = StageRecord(STAGE, "summary")
    cf = read_stage(STAGE, "configs")["numbers"]["configs"]
    dk = read_stage(STAGE, "dkin")["numbers"]["dkin"]
    res, ok = {}, True
    g = np.random.default_rng(15)
    for s in SAMPLES:
        Dk = dk[s]["D_formula_onesided_bprp"]
        res[s] = {}
        for cut in (30, 40):
            res[s][cut] = {}
            for k, cfg in enumerate(("A", "C")):
                d = np.array(cf[s][str(cut)][cfg]["d_eq"]); D = float(np.linalg.norm(d))
                res[s][cut][cfg] = dict(D=D, D_over_Dkin=D / Dk, angle_to_CMB=cf[s][str(cut)][cfg]["angle_to_CMB"],
                                        l=cf[s][str(cut)][cfg]["l"], b=cf[s][str(cut)][cfg]["b"], models={})
                for m in ("lcdm", "data"):
                    x = np.concatenate([np.load(OUT / f"mocks_{s}_{cut}_{m}_{c}.npy") for c in range(4)])[:, k, :]
                    mu = x.mean(0); cov = np.cov(x.T); Lc = np.linalg.cholesky(cov)
                    if cfg == "A":
                        z = (mu - Dk * cmb_eq) / (x.std(0, ddof=1) / np.sqrt(len(x)))
                        ok &= bool(np.all(np.abs(z) < 3))
                    hits = sum(int(np.sum(np.linalg.norm(mu + g.standard_normal((2_000_000, 3)) @ Lc.T, axis=1) >= D))
                               for _ in range(10))
                    pa = max(hits, 0.5) / 2e7
                    res[s][cut][cfg]["models"][m] = dict(sigma_D=float(np.sqrt((d / D) @ cov @ (d / D))),
                                                         p=pa, sigma=float(norm.isf(pa)), hits=hits)
    rec.check("5. Quaia null mocks, config A: unbiased (all samples, cuts, models)", ok)
    rec.number("results", res)
    # figure: Mittal reproduction and our configs vs cut
    mt = read_stage(STAGE, "mittal")["numbers"]["mittal_model"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for j, s in enumerate(SAMPLES):
        a = ax[j]
        a.plot(CUTS, [MITTAL[s][c] for c in CUTS], "s", color="#7a7975", ms=8, label="Mittal et al. (Fig. 6)")
        err = lambda c, k: np.sqrt(np.mean(np.square(cf[s][str(c)][k]["sigma_formal"]))) * 1e3
        a.errorbar(CUTS, [mt[s][str(c)]["D"] * 1e3 for c in CUTS], yerr=[err(c, "A") for c in CUTS], fmt="o-",
                   color="#0b0b0b", capsize=3, label="ours, Mittal's model (A)")
        a.errorbar(np.array(CUTS) + 0.8, [cf[s][str(c)]["C"]["D"] * 1e3 for c in CUTS], yerr=[err(c, "C") for c in CUTS],
                   fmt="o--", color="#eb6834", capsize=3, label="ours, + ℓ = 2, 3 (C)")
        a.plot(CUTS, [cf[s][str(c)]["A+scan"]["D"] * 1e3 for c in CUTS], "o:", color="#2a78d6", label="ours, + unWISE scan")
        a.axhline(dk[s]["D_formula_onesided_bprp"] * 1e3, color="k", ls=":", lw=1)
        a.text(41, dk[s]["D_formula_onesided_bprp"] * 1e3, " D_kin", va="center")
        a.set_xlabel("Galactic cut |b| ≥ (deg)"); a.set_ylabel("dipole amplitude (× 10⁻³)\nerror bars: formal 1σ per component")
        a.set_title(f"Quaia {s} (G < {SAMPLES[s]})", loc="left"); a.grid(alpha=0.3); a.set_xlim(-3, 47)
        if j == 0:
            a.legend(frameon=False, fontsize=9, loc="upper right")
    plt.tight_layout(); plt.savefig(OUT / "quaia_dipole.png", dpi=130); plt.close()
    rec.write()
else:
    sys.exit(f"unknown part {part}")
