"""
Stage s11: clustering models for the mocks. Project Step 4, first half.

Parts (each writes summary_<part>.json):
  pixwin      the HEALPix pixel window, computed directly (healpy's tabulated
              file cannot be downloaded here); compared with a simulation
  theory      CAMB spectrum for the quasars (Planck 2018; N(z) read from S21
              Fig. 3; Tiwari et al.'s best-fit bias b(z)), with redshift-space
              distortions, lensing magnification and nonlinear power; plus an
              independent exact linear calculation at low l, which also gives
              l = 1 (CAMB starts at l = 2). Needs no catalog data.
              Slow at the accuracy needed (several minutes and ~2 GB of memory);
              CAMB's default accuracy is 10-20% off at l < 40 for this window.
  lognormal   tests of the lognormal sky generator on the full sky
  noise       100 Poisson-only mocks: the shot-noise level of the measured
              spectrum, processed exactly as the data
  fit         fits two clustering models to the data's measured spectrum,
              both by forward modelling (mocks processed exactly as the data):
                lcdm: the CAMB shape times one amplitude fitted at l >= 26
                      (as Tiwari et al. fit only small scales)
                data: the same, times a free factor in each multipole band,
                      so that the low-l power is whatever the data show
  check       200 mocks per model: does each model reproduce the data's
              measured spectrum?

Revision after the first run (kept as results/s11_clustering/v1_no_subpixel_*):
the first models had no clustering on scales smaller than a pixel, and the
lcdm model failed check 5 with excess power rising toward l = 191. Clustering
at l > 191 does not vanish when sources are counted in pixels: it adds
pixel-to-pixel scatter beyond Poisson (super-Poissonian counts). Its size is
predicted, not fitted: the pixel variance from the theory spectrum at l > 191
(times the fitted amplitude; extended beyond l = 260 as a power law), smoothed
by the pixel window. Every clustered mock now multiplies each pixel's mean by
an independent Gamma variable with that variance.
  summary     checks, numbers, figure

Checks (set before running)
  1. The N(z) read from S21 Fig. 3 has S21's mean redshift, 1.2 (within 0.05).
  2. CAMB (density term only, linear) agrees with the exact calculation within
     3% for l = 2-40 at the accuracy setting used.
  3. Lognormal generator, full sky: the mean recovered spectrum matches the
     input in every band within 3 standard errors; delta > -1 everywhere;
     the clustering-dipole power C_1 is reproduced within 3 standard errors.
  4. The data-driven model reproduces the data's measured spectrum in every
     band within 0.2 mock standard deviations (convergence of the fit).
  5. The lcdm model fits the data's spectrum at l >= 26 (chi-square p > 0.01).
Recorded, not checks: the theory C_1 against Tiwari et al.'s 0.91e-6; the
data's excess power at low l relative to lcdm, band by band.

The spectra are of the fractional overdensity; the data's spectrum is measured
on the S22 footprint after dividing out the fitted monopole, dipole and
multiplicative ecliptic trend (dipole.clustering.residual_map). Mocks are
processed identically, so mode loss, mask coupling, pixelization and shot
noise affect data and mocks alike.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, healpy as hp
import config
from dipole import load_mask, load_inputs, unit_vector, cos_theta_map, ecliptic_latitude_map, NPIX
from dipole.mocks import expected_counts
from dipole.clustering import (nz_s21, bias_tiwari, camb_cls, exact_linear_cls, gaussian_cls_for_lognormal,
                               lognormal_delta, pseudo_cl, band, BANDS)
from dipole.checks import StageRecord, read_stage

STAGE = "s11_clustering"
part = sys.argv[1] if len(sys.argv) > 1 else "summary"
LMAX = 3 * config.NSIDE - 1                     # 191
CAMB_ACC = dict(AccuracyBoost=3, lSampleBoost=1, lAccuracyBoost=1)   # the highest that fits in 7 GB; see AI log
CAMB_LMAX = 260                                  # well above l = 191: CAMB values near its own l_max are unreliable
X_FOR_MAGNIFICATION = config.X_S22              # dlog10N/dm = 0.4 x
HIGH = [i for i, (a, b) in enumerate(BANDS) if a >= 26]


def pixwin():
    return np.load(config.RESULTS / STAGE / "pixwin.npy")[: LMAX + 1]


def gen_spectrum(cl):
    """Spectrum of the lognormal field that gives `cl` after pixelization."""
    w = pixwin()
    return gaussian_cls_for_lognormal(cl[: LMAX + 1] * w ** 2, LMAX)


def base_lambda(I, mask, D):
    return expected_counts(cos_theta_map(unit_vector()), np.abs(ecliptic_latitude_map()), mask,
                           N0=I["ecl_intercept_count"], s=I["ecl_frac_slope_per_deg"], D=D)


def subpixel_variance(A, lmax_ext=1500):
    """Pixel variance of the overdensity from clustering at l > 191 (theory x A;
    beyond l = 255 extended as a power law fitted at l = 150-255)."""
    from dipole.clustering import pixel_window
    cl = np.load(config.RESULTS / STAGE / "theory_cls.npy")
    L = np.arange(len(cl))
    p = np.polyfit(np.log(L[150:255]), np.log(cl[150:255]), 1)
    ll = np.arange(lmax_ext + 1)
    C = np.zeros(lmax_ext + 1); C[:250] = cl[:250]; C[250:] = np.exp(np.polyval(p, np.log(ll[250:])))
    w = pixel_window(lmax=lmax_ext)
    return float(np.sum((2 * ll[LMAX + 1:] + 1) / (4 * np.pi) * A * C[LMAX + 1:] * w[LMAX + 1:] ** 2)), float(p[0])


def subpixel_factor(pixvar, n, rng):
    """Independent per-pixel multiplier, mean 1, variance pixvar (Gamma)."""
    k = 1.0 / pixvar
    return rng.gamma(k, 1.0 / k, n)


def mock_bands(clg, lam, mask, trend, seeds, pixvar=0.0):
    """Measured-spectrum bands of clustered (or, clg=None, Poisson) mocks."""
    out = []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        mu = lam if clg is None else lam * (1 + lognormal_delta(clg, LMAX, rng))
        if pixvar > 0:
            mu = mu * subpixel_factor(pixvar, mu.size, rng)
        out.append(band(pseudo_cl(rng.poisson(mu).astype(float), mask, trend, LMAX)))
    return np.array(out)


def theory_band_pix(cl):
    w = pixwin()
    return band(cl[: LMAX + 1] * w ** 2)


def data_bands():
    from dipole.catalog import load_catalog, galactic_pixels, s22_selection, count_map
    cat = load_catalog(("ra", "dec", "w1", "w1cov"))
    sel = s22_selection(cat)
    cnt = count_map(galactic_pixels(cat["ra"][sel], cat["dec"][sel])).astype(float)
    mask = load_mask("s22")
    return band(pseudo_cl(cnt, mask, np.abs(ecliptic_latitude_map()), LMAX)), cnt


# ================================================================= parts
if part == "pixwin":
    from dipole.clustering import pixel_window, random_alm
    rec = StageRecord(STAGE, "pixwin")
    w = pixel_window()
    np.save(rec.dir / "pixwin.npy", w)
    R = []
    for i in range(20):
        a = random_alm(np.ones(LMAX + 1), LMAX, np.random.default_rng(19_000_000 + i))
        A = hp.ud_grade(hp.alm2map(a, 512, lmax=LMAX, pixwin=False), config.NSIDE)
        R.append(hp.alm2cl(hp.map2alm(A, lmax=LMAX, iter=10)) / hp.alm2cl(a))
    R = np.mean(R, 0)
    rec.number("w2_direct_vs_simulation", {l: [float(w[l] ** 2), float(R[l])] for l in (10, 50, 100, 150, 191)})
    rec.note("The direct calculation is HEALPix's standard (direction-averaged) pixel window. The simulation, which "
             "averages a band-limited N_side = 512 map down to 64, gives 0.4-5% more power at l >= 50 because "
             "averaging over a non-circular pixel also moves power between multipoles. Irrelevant for the dipole "
             "(w^2 = 0.9974 at l = 10, agreeing to 4 digits); the mocks' amplitude is fitted to the data.")
    rec.write()

elif part == "theory":
    import camb
    from camb.sources import SplinedSourceWindow
    from dipole.clustering import PLANCK18
    rec = StageRecord(STAGE, "theory")
    zg, nz, zmean, _ = nz_s21("table")
    zg2, nz2, zmean2, info2 = nz_s21("fit")
    bz = bias_tiwari(zg)
    rec.number("nz", dict(mean_z_table=zmean, mean_z_param_fit=zmean2, param_fit=info2))
    rec.check("1. N(z) from S21 Fig. 3 has S21's mean redshift 1.2", abs(zmean - 1.2) < 0.05, f"{zmean:.3f}")

    def run(all_terms, nonlinear, lmax=CAMB_LMAX):
        pars = camb.set_params(**PLANCK18, lmax=lmax)
        pars.Want_CMB = False
        pars.NonLinear = camb.model.NonLinear_both if nonlinear else camb.model.NonLinear_none
        st = pars.SourceTerms
        for k in ("counts_redshift", "counts_lensing", "counts_velocity", "counts_radial",
                  "counts_timedelay", "counts_ISW", "counts_potential"):
            setattr(st, k, all_terms and k in ("counts_redshift", "counts_lensing"))
        st.counts_evolve = False
        pars.SourceWindows = [SplinedSourceWindow(z=zg[1:], W=nz[1:], bias_z=bz[1:],
                                                  dlog10Ndm=0.4 * X_FOR_MAGNIFICATION if all_terms else 0.0)]
        pars.set_accuracy(**CAMB_ACC)
        r = camb.get_results(pars)
        return r.get_source_cls_dict(raw_cl=True)["W1xW1"][: lmax + 1].copy(), r

    cl_dens, r = run(False, False)
    ex = exact_linear_cls(r, zg, nz, bz, ells=(1, 2, 3, 4, 5, 10, 20, 40), nk=6000, kmax=0.5)
    ratios = {l: ex[l] / cl_dens[l] for l in ex if l >= 2}
    rec.number("exact_over_camb_density_linear", ratios)
    rec.check("2. CAMB (density, linear) = exact calculation within 3%, l = 2-40",
              all(abs(v - 1) < 0.03 for v in ratios.values()),
              ", ".join(f"l={l}: {v:.3f}" for l, v in ratios.items()))
    cl_full, _ = run(True, True)
    cl_full[0] = 0.0
    cl_full[1] = ex[1]                       # exact linear density-only C_1
    np.save(rec.dir / "theory_cls.npy", cl_full)
    rec.number("C1_exact", ex[1])
    rec.number("C1_Tiwari", 0.91e-6)
    rec.number("clustering_dipole_rms_theory", float(np.sqrt(9 * ex[1] / (4 * np.pi))))
    rec.number("full_over_density_only_l2_10_40", [cl_full[l] / cl_dens[l] for l in (2, 10, 40)])
    rec.note("N(z) is read by eye from S21 Fig. 3 (no table is published); only the spectrum's shape "
             "depends on it, because the amplitude is fitted to the data in part 'fit'.")
    rec.note("Check 2: at CAMB's default accuracy the exact calculation was 21% below CAMB at l = 2 and 9% at "
             "l = 10-40. Raising CAMB's accuracy moves CAMB toward the exact result (AccuracyBoost 2: 6% at l = 2; "
             "3: 3%). Accuracy 4 runs out of memory (7 GB), so 3 is used; the remaining difference is 3.0% at l = 2, "
             "1.6% at l = 3 and <= 0.8% from l = 4 on. The check's 3% tolerance is missed at l = 2 only. This does not "
             "matter for the mocks: the data's l = 2 power is ~25 times the theory value and the amplitude is fitted "
             "at l >= 26.")
    rec.write()

elif part == "lognormal":
    rec = StageRecord(STAGE, "lognormal")
    cl = np.load(rec.dir / "theory_cls.npy")[: LMAX + 1]
    res = {}
    ok = True
    for amp in (1.0, 20.0):
        target = amp * cl
        clg = gaussian_cls_for_lognormal(target, LMAX)
        P, mins, means = [], [], []
        for i in range(200):
            dlt = lognormal_delta(clg, LMAX, np.random.default_rng(14_000_000 + i))
            P.append(hp.anafast(dlt, lmax=LMAX)); mins.append(dlt.min()); means.append(dlt.mean())
        P = np.array(P)
        bands_m = np.array([band(p, [(1, 1)] + BANDS) for p in P])
        tb = band(target, [(1, 1)] + BANDS)
        z = (bands_m.mean(0) - tb) / (bands_m.std(0, ddof=1) / np.sqrt(len(P)))
        res[f"x{amp:g}"] = dict(z_by_band=z.tolist(), min_delta=float(min(mins)),
                                mean_delta=float(np.mean(means)), C1_ratio=float(bands_m[:, 0].mean() / tb[0]))
        ok &= bool(np.all(np.abs(z) < 3) and min(mins) > -1)
    rec.number("tests", res)
    rec.check("3. lognormal generator reproduces the input spectrum (incl. C_1), delta > -1",
              ok, "; ".join(f"{k}: max|z| {np.max(np.abs(v['z_by_band'])):.2f}, min delta {v['min_delta']:.3f}"
                            for k, v in res.items()))
    # 3b, added after check 3 failed in the band l = 121-191: is the shortfall the lognormal step,
    # or HEALPix's spectrum measurement near its resolution limit? Compare with Gaussian maps
    # of the same input put through the same map -> spectrum round trip (same seeds).
    from dipole.clustering import random_alm
    ok3b, res3b = True, {}
    for amp in (1.0, 20.0):
        target = amp * cl
        clg = gaussian_cls_for_lognormal(target, LMAX)
        Lg, Gg = [], []
        for i in range(200):
            Lg.append(band(hp.anafast(lognormal_delta(clg, LMAX, np.random.default_rng(14_000_000 + i)), lmax=LMAX), BANDS))
            g = hp.alm2map(random_alm(target, LMAX, np.random.default_rng(14_500_000 + i)), config.NSIDE, lmax=LMAX, pixwin=False)
            Gg.append(band(hp.anafast(g, lmax=LMAX), BANDS))
        Lg, Gg = np.array(Lg), np.array(Gg)
        z = (Lg.mean(0) - Gg.mean(0)) / np.sqrt(Lg.var(0, ddof=1) / len(Lg) + Gg.var(0, ddof=1) / len(Gg))
        res3b[f"x{amp:g}"] = dict(z_lognormal_vs_gaussian=z.tolist(),
                                 gaussian_roundtrip_over_input_last_band=float(Gg.mean(0)[-1] / band(target, BANDS)[-1]))
        ok3b &= bool(np.all(np.abs(z) < 3))
    rec.number("test_3b", res3b)
    rec.check("3b (added after 3 failed). lognormal maps = Gaussian maps through the same round trip, every band",
              ok3b, "; ".join(f"{k}: max|z| {np.max(np.abs(v['z_lognormal_vs_gaussian'])):.2f}, Gaussian round trip "
                              f"recovers {v['gaussian_roundtrip_over_input_last_band']:.3f} of the input at l = 121-191"
                              for k, v in res3b.items()))
    rec.note("Check 3 fails only in the band l = 121-191, where the recovered power is ~3% below the input. Plain "
             "Gaussian maps lose the same 3% (check 3b): HEALPix's spectrum measurement (anafast without iteration) "
             "is inaccurate near its resolution limit (l near 3 N_side). Data and mocks go through the same measurement, "
             "so this does not bias the fitted clustering models. Check 3b passes at the realistic amplitude (x1). At 20 times "
             "the theory amplitude (a stress test; delta reaches -0.85) the lognormal maps have extra power at l = 121-191 "
             "only, from the nonlinear transform spilling power past the band limit. The fitted models are near x1 except "
             "at a few low-l bands, and the highest band does not affect the dipole.")
    rec.write()

elif part == "noise":
    rec = StageRecord(STAGE, "noise")
    I = load_inputs(); mask = load_mask("s22"); ae = np.abs(ecliptic_latitude_map())
    lam = base_lambda(I, mask, 0.0148)
    B = mock_bands(None, lam, mask, ae, [15_000_000 + i for i in range(100)])
    np.save(rec.dir / "noise_bands.npy", B)
    rec.number("noise_band_mean", B.mean(0).tolist())
    rec.write()

elif part == "fit":
    rec = StageRecord(STAGE, "fit")
    I = load_inputs(); mask = load_mask("s22"); ae = np.abs(ecliptic_latitude_map())
    lam = base_lambda(I, mask, 0.0148)
    Pd, _ = data_bands()
    N = np.load(rec.dir / "noise_bands.npy").mean(0)
    cl_th = np.load(rec.dir / "theory_cls.npy")[: LMAX + 1]
    tb = theory_band_pix(cl_th)
    # lcdm: one amplitude from l >= 26
    A, hist = 1.0, []
    for it in range(4):
        pv = subpixel_variance(A)[0]
        M = mock_bands(gen_spectrum(A * cl_th), lam, mask, ae, [16_000_000 + 1000 * it + i for i in range(150)], pv).mean(0)
        A *= np.sum(Pd[HIGH] - N[HIGH]) / np.sum(M[HIGH] - N[HIGH])
        hist.append(A)
    pixvar, slope = subpixel_variance(A)
    rec.number("subpixel_pixel_variance", pixvar)
    rec.number("subpixel_white_power", pixvar * hp.nside2pixarea(config.NSIDE))
    rec.number("theory_slope_l150_255", slope)
    rec.number("lcdm_amplitude_iterations", hist)
    cl_lcdm = A * cl_th
    # data-driven: per-band factor on top
    edges = BANDS
    r = np.ones(len(edges)); rh = []
    for it in range(10):
        cl_d = cl_lcdm.copy()
        for j, (a, b) in enumerate(edges):
            cl_d[a:b + 1] *= r[j]
        M = mock_bands(gen_spectrum(cl_d), lam, mask, ae, [17_000_000 + 1000 * it + i for i in range(200)], pixvar).mean(0)
        r = np.maximum(0.0, r + (Pd - M) / (A * tb))
        rh.append(r.tolist())
    cl_data = cl_lcdm.copy()
    for j, (a, b) in enumerate(edges):
        cl_data[a:b + 1] *= r[j]
    np.save(rec.dir / "cl_lcdm.npy", cl_lcdm)
    np.save(rec.dir / "cl_data.npy", cl_data)
    rec.number("lcdm_amplitude", A)
    rec.number("data_band_factors_iterations", rh)
    rec.number("data_band_factors", r.tolist())
    rec.write()

elif part == "check":
    rec = StageRecord(STAGE, "check")
    I = load_inputs(); mask = load_mask("s22"); ae = np.abs(ecliptic_latitude_map())
    lam = base_lambda(I, mask, 0.0148)
    Pd, _ = data_bands()
    out = {}
    pixvar = read_stage(STAGE, "fit")["numbers"]["subpixel_pixel_variance"]
    for name in ("lcdm", "data"):
        cl = np.load(rec.dir / f"cl_{name}.npy")
        B = mock_bands(gen_spectrum(cl), lam, mask, ae, [18_000_000 + i for i in range(200)], pixvar)
        np.save(rec.dir / f"check_bands_{name}.npy", B)
        out[name] = dict(mean=B.mean(0).tolist(), sd=B.std(0, ddof=1).tolist())
    rec.number("data_bands", Pd.tolist())
    rec.number("mock_bands", out)
    rec.write()

elif part == "summary":
    from scipy.stats import chi2
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rec = StageRecord(STAGE, "summary")
    th = read_stage(STAGE, "theory"); ln = read_stage(STAGE, "lognormal"); fit = read_stage(STAGE, "fit")
    ck = read_stage(STAGE, "check")["numbers"]
    Pd = np.array(ck["data_bands"])
    m = {k: np.array(v["mean"]) for k, v in ck["mock_bands"].items()}
    s = {k: np.array(v["sd"]) for k, v in ck["mock_bands"].items()}
    zdat = (Pd - m["data"]) / s["data"]
    rec.check("4. data-driven model reproduces the data's spectrum (every band within 0.2 sd)",
              np.all(np.abs(zdat) < 0.2), f"max |offset| {np.max(np.abs(zdat)):.2f} sd")
    zl = (Pd - m["lcdm"]) / s["lcdm"]
    c2 = float(np.sum(zl[HIGH] ** 2)); p = float(chi2.sf(c2, len(HIGH)))
    rec.check("5. lcdm model fits the data at l >= 26 (revised model, with sub-pixel clustering)", p > 0.01,
              f"chi2 {c2:.1f} for {len(HIGH)} bands, p = {p:.3f}")
    # The first run's failed check 5 is kept on record. Its outputs (v1_no_subpixel_*) are a historical record
    # that the current code does not regenerate, so a fresh run uses the value recorded from that run.
    v1f = rec.dir / "v1_no_subpixel_summary.json"
    v1_detail = ([c for c in __import__("json").loads(v1f.read_text())["checks"] if c["name"].startswith("5.")][0]["detail"]
                 if v1f.exists() else "chi2 23.8 for 4 bands (recorded from the first run, Nov 18; see AI log entry 37)")
    rec.checks.append(dict(name="5 (first run, without sub-pixel clustering; superseded)", passed=False,
                           detail=v1_detail))
    rec.number("subpixel_pixel_variance", fit["numbers"]["subpixel_pixel_variance"])
    rec.note("Check 4 fails only at l = 5 and l = 6-7, where the data's power is below the shot-noise level, so even "
             "zero clustering in those bands cannot match them (power cannot be negative); every other band is within "
             "0.2 sd. Check 5 failed in the first run (no sub-pixel clustering): the data had excess power rising toward "
             "l = 191, of the size predicted for clustering at l > 191 counted in pixels. The models were revised to "
             "include it (see the stage docstring); the first run is kept as v1_no_subpixel_*.")
    rec.number("lcdm_offsets_by_band_in_sd", dict(zip([f"{a}-{b}" for a, b in BANDS], zl.tolist())))
    rec.number("data_over_lcdm_power_by_band", dict(zip([f"{a}-{b}" for a, b in BANDS], fit["numbers"]["data_band_factors"])))
    rec.number("lcdm_amplitude", fit["numbers"]["lcdm_amplitude"])
    rec.number("C1", dict(theory=th["numbers"]["C1_exact"], tiwari=0.91e-6,
                          lcdm_model=float(np.load(rec.dir / "cl_lcdm.npy")[1])))
    # figure
    L = np.array([(a + b) / 2 for a, b in BANDS])
    N = np.array(read_stage(STAGE, "noise")["numbers"]["noise_band_mean"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(L, (Pd - N) * 1e6, yerr=s["lcdm"] * 1e6, fmt="o", color="#0b0b0b", label="CatWISE (shot noise removed)", zorder=3)
    for k, c, lab in [("lcdm", "#2a78d6", "ΛCDM shape, amplitude fitted at ℓ ≥ 26"), ("data", "#eb6834", "data-driven")]:
        ax.plot(L, (m[k] - N) * 1e6, "-", color=c, lw=2, label=f"mocks: {lab}")
    ax.set_xscale("log"); ax.set_xlabel("multipole ℓ (band centres)"); ax.set_ylabel("measured power − shot noise  (× 10⁻⁶)")
    ax.axhline(0, color="#999", lw=0.8); ax.legend(frameon=False); ax.grid(alpha=0.3)
    ax.set_title("Angular power of CatWISE quasars on the S22 footprint", loc="left")
    plt.tight_layout(); plt.savefig(rec.dir / "spectrum.png", dpi=130); plt.close()
    rec.write()
else:
    sys.exit(f"unknown part {part}")
