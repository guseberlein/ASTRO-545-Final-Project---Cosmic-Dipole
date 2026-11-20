"""
Stage s12: null mocks with clustering, and the four fit configurations.
Project Step 4, second half.

Null hypothesis: the sky is the kinematic dipole (D_kin, CMB direction), the
multiplicative ecliptic pattern (straight line in |ecliptic latitude|, as
measured in s02), clustering, and Poisson noise, on the S22 footprint.

Clustering models (from s11):
  poisson   no clustering (the prototype's mocks)
  lcdm      lognormal, CAMB shape, amplitude fitted to the data at l >= 26,
            including the theory clustering dipole C_1, plus the per-pixel
            scatter from clustering at l > 191 (s11)
  data      lognormal, the same but with each low-l band's power set by the
            data (the excess at l = 2 and 4 included)

Fit configurations (dipole always fitted, with a monopole):
  A  + ecliptic template                     (the prototype's fit A; S22's choice)
  B  no ecliptic template                    (the prototype's fit B)
  C  + ecliptic template + l = 2, 3 templates
  D  l = 2, 3 templates, no ecliptic template

Parts:
  templates         noiseless tests of the configurations
  mocks <model> <c> 1000 mocks per model in chunks c = 0..3 of 250.
                    Poisson draws use the prototype's seeds 8_000_000 + i, so
                    'poisson' mocks 0-499 are the prototype's s08 mocks.
  summary           checks, null distributions, figure

Checks (set before running)
  1. Noiseless: config C recovers the dipole exactly from a map with dipole,
     ecliptic pattern and an l = 2, 3 pattern; config D does so without the
     ecliptic pattern; A and C agree exactly when no l = 2, 3 pattern exists.
  2. 'poisson' mocks 0-499 with config A reproduce the prototype's fit A
     dipole vectors (s08) to 1e-12.
  3. With clustering (lcdm, data), the configs that model the ecliptic
     pattern (A, C) are unbiased: the mean dipole vector equals D_kin toward
     the CMB within 3 standard errors per component.
Recorded: per model and config, the spread of the dipole components, the
mean amplitude, the bias of the configs without the ecliptic template, and
how often a null mock reaches S22's amplitude 0.0148 (config A only; the
real data are measured in all four configs in Step 5).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, healpy as hp
import config
from dipole import (load_mask, load_inputs, unit_vector, cos_theta_map, ecliptic_latitude_map,
                    fit_multipoles_mult, NPIX)
from dipole.mocks import expected_counts, draw_mock
from dipole.clustering import gaussian_cls_for_lognormal, lognormal_delta, ylm_templates
from dipole.checks import StageRecord, read_stage

STAGE = "s12_nullmocks"
part = sys.argv[1] if len(sys.argv) > 1 else "summary"
LMAX = 3 * config.NSIDE - 1
CONFIGS = ["A", "B", "C", "D"]
N_PER_CHUNK, N_CHUNKS = 250, 4

I = load_inputs()
D = I["D_kin_from_published"]
mask = load_mask("s22")
ae = np.abs(ecliptic_latitude_map())
dhat = unit_vector()
Y, _ = ylm_templates((2, 3))


def fit_all(c):
    return [fit_multipoles_mult(c, mask, ae),
            fit_multipoles_mult(c, mask),
            fit_multipoles_mult(c, mask, ae, Y),
            fit_multipoles_mult(c, mask, None, Y)]


def lam0(ecl=True):
    return expected_counts(cos_theta_map(dhat), ae, mask, N0=I["ecl_intercept_count"],
                           s=I["ecl_frac_slope_per_deg"], D=D, ecliptic=ecl)


if part == "templates":
    rec = StageRecord(STAGE, "templates")
    rng = np.random.default_rng(5)
    q = rng.normal(0, 0.01, Y.shape[0])
    # exact model of config C: a(1+s t)(1 + d.n + q.Y)
    base = I["ecl_intercept_count"] * (1 + I["ecl_frac_slope_per_deg"] * ae)
    mC = base * (1 + D * cos_theta_map(dhat) + q @ Y); mC[mask] = 0
    rC = fit_multipoles_mult(mC, mask, ae, Y)
    mD = I["ecl_intercept_count"] * (1 + D * cos_theta_map(dhat) + q @ Y); mD[mask] = 0
    rD = fit_multipoles_mult(mD, mask, None, Y)
    mA = lam0()
    rA, rC0 = fit_multipoles_mult(mA, mask, ae), fit_multipoles_mult(mA, mask, ae, Y)
    eC, eD = np.linalg.norm(rC["d"] - D * dhat), np.linalg.norm(rD["d"] - D * dhat)
    eA = np.linalg.norm(rA["d"] - rC0["d"])
    rec.check("1. noiseless: C and D recover the dipole exactly; A = C without an l = 2, 3 pattern",
              eC < 1e-10 and eD < 1e-10 and eA < 1e-10, f"errors {eC:.1e}, {eD:.1e}; A vs C {eA:.1e}")
    # noiseless biases of B and D (ecliptic pattern present, not modelled)
    rB, rD2 = fit_multipoles_mult(mA, mask), fit_multipoles_mult(mA, mask, None, Y)
    ang = lambda r: float(np.degrees(np.arccos(np.clip(r["d"] @ dhat / r["amp"], -1, 1))))
    rec.number("noiseless_amp_over_D", dict(A=rA["amp"] / D, B=rB["amp"] / D, C=rC0["amp"] / D, D=rD2["amp"] / D))
    rec.number("noiseless_angle_deg", dict(A=ang(rA), B=ang(rB), C=ang(rC0), D=ang(rD2)))
    rec.write()

elif part == "mocks":
    model, chunk = sys.argv[2], int(sys.argv[3])
    rec = StageRecord(STAGE, f"mocks_{model}_{chunk}")
    lam = lam0()
    if model != "poisson":
        cl = np.load(config.RESULTS / "s11_clustering" / f"cl_{model}.npy")[: LMAX + 1]
        w = np.load(config.RESULTS / "s11_clustering" / "pixwin.npy")[: LMAX + 1]
        clg = gaussian_cls_for_lognormal(cl * w ** 2, LMAX)
        pixvar = read_stage("s11_clustering", "fit")["numbers"]["subpixel_pixel_variance"]
    out = np.zeros((N_PER_CHUNK, 4, 3))
    for j in range(N_PER_CHUNK):
        i = chunk * N_PER_CHUNK + j
        if model == "poisson":
            mu = lam
        else:
            rf = np.random.default_rng(13_000_000 + i)
            mu = lam * (1 + lognormal_delta(clg, LMAX, rf))
            k = 1.0 / pixvar
            mu = mu * rf.gamma(k, 1.0 / k, mu.size)            # sub-pixel clustering (see s11)
        c = draw_mock(mu, np.random.default_rng(8_000_000 + i))
        out[j] = [f["d"] for f in fit_all(c)]
    np.save(rec.dir / f"{model}_{chunk}.npy", out)
    rec.number("n", N_PER_CHUNK)
    rec.write()

elif part == "summary":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rec = StageRecord(STAGE, "summary")
    tm = read_stage(STAGE, "templates")
    R = {m: np.concatenate([np.load(rec.dir / f"{m}_{c}.npy") for c in range(N_CHUNKS)])
         for m in ("poisson", "lcdm", "data")}
    s8 = np.vstack([np.load(config.RESULTS / "s08_main" / f"A_{c}.npy") for c in range(4)])[:, :3]
    err = float(np.max(np.abs(R["poisson"][:500, 0, :] - s8)))
    rec.check("2. poisson mocks 0-499, config A = prototype s08 fit A", err < 1e-12, f"max difference {err:.1e}")
    stats, ok3 = {}, True
    for m, arr in R.items():
        stats[m] = {}
        for k, cfg in enumerate(CONFIGS):
            d = arr[:, k, :]
            n = len(d)
            mean = d.mean(0); se = d.std(0, ddof=1) / np.sqrt(n)
            amp = np.linalg.norm(d, axis=1)
            cov = np.cov(d.T)
            stats[m][cfg] = dict(mean_vec_over_D=(mean / D).tolist(), mean_par_over_D=float(mean @ dhat / D),
                                 se_par_over_D=float(np.sqrt(dhat @ cov @ dhat / n) / D),
                                 sigma_components=np.sqrt(np.diag(cov)).tolist(),
                                 sigma_rms=float(np.sqrt(np.trace(cov) / 3)),
                                 mean_amp=float(amp.mean()), amp_percentiles_50_95_99=np.percentile(amp, [50, 95, 99]).tolist(),
                                 frac_amp_ge_0148=float(np.mean(amp >= 0.0148)), cov=cov.tolist(), mean=mean.tolist())
            if m != "poisson" and cfg in ("A", "C"):
                z = (mean - D * dhat) / se
                stats[m][cfg]["unbiased_z"] = z.tolist()
                ok3 &= bool(np.all(np.abs(z) < 3))
    rec.check("3. with clustering, configs A and C are unbiased (mean vector = D_kin toward the CMB)", ok3,
              "; ".join(f"{m} {c}: z = {np.round(stats[m][c]['unbiased_z'], 1).tolist()}" for m in ("lcdm", "data") for c in ("A", "C")))
    rec.number("stats", stats)
    # Preview (not a check): how often does a Gaussian fitted to each null distribution of dipole vectors reach
    # S22's amplitude 0.0148? Gaussian validity is checked on the mocks' own 95th and 99th percentiles.
    from scipy.stats import norm
    prev = {}
    for m in R:
        prev[m] = {}
        for k, cfg in enumerate(CONFIGS):
            mu, cov = np.array(stats[m][cfg]["mean"]), np.array(stats[m][cfg]["cov"])
            Lc = np.linalg.cholesky(cov)
            g = np.random.default_rng(99)
            hits, n, amps = 0, 0, []
            for _ in range(10):
                z = mu + g.standard_normal((2_000_000, 3)) @ Lc.T
                a = np.linalg.norm(z, axis=1); hits += int(np.sum(a >= 0.0148)); n += a.size
                if not amps: amps = np.percentile(a, [95, 99]).tolist()
            pg = max(hits, 0.5) / n
            emp = np.percentile(np.linalg.norm(R[m][:, k, :], axis=1), [95, 99])
            prev[m][cfg] = dict(p_gauss=pg, sigma_one_sided=float(norm.isf(pg)), hits=hits, draws=n,
                                gauss_p95_p99=amps, mock_p95_p99=emp.tolist())
    rec.number("preview_S22_amplitude_vs_null", prev)
    rec.number("spread_relative_to_poisson_A", {m: {c: stats[m][c]["sigma_rms"] / stats["poisson"]["A"]["sigma_rms"]
                                                     for c in CONFIGS} for m in stats})
    # figure: component spread and amplitude distributions
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    cols = {"poisson": "#7a7975", "lcdm": "#2a78d6", "data": "#eb6834"}
    x = np.arange(4)
    for j, m in enumerate(stats):
        ax[0].bar(x + (j - 1) * 0.27, [stats[m][c]["sigma_rms"] * 1e3 for c in CONFIGS], 0.25, color=cols[m], label=m)
    ax[0].set_xticks(x); ax[0].set_xticklabels(["A: + ecliptic", "B: dipole only", "C: + ecliptic + ℓ=2,3", "D: + ℓ=2,3"])
    ax[0].set_ylabel("spread of each dipole component  (× 10⁻³)"); ax[0].legend(frameon=False); ax[0].grid(axis="y", alpha=0.3)
    bins = np.linspace(0, 0.022, 45)
    for m in stats:
        for k, ls in [(0, "-"), (2, "--")]:
            ax[1].hist(np.linalg.norm(R[m][:, k, :], axis=1), bins=bins, histtype="step", color=cols[m], ls=ls, lw=1.8,
                       label=f"{m}, config {CONFIGS[k]}")
    ax[1].axvline(D, color="k", lw=1); ax[1].axvline(0.0148, color="k", ls=":", lw=1)
    tr = ax[1].get_xaxis_transform()
    ax[1].text(0.0148, 1.01, "S22", ha="center", va="bottom", transform=tr); ax[1].text(D, 1.01, "D_kin", ha="center", va="bottom", transform=tr)
    ax[1].set_xlim(0, 0.022)
    ax[1].set_xlabel("fitted dipole amplitude |d|  (null mocks)"); ax[1].set_ylabel("mocks (of 1000)")
    ax[1].legend(frameon=False, fontsize=8)
    plt.tight_layout(); plt.savefig(rec.dir / "null_distributions.png", dpi=130); plt.close()
    rec.write()
else:
    sys.exit(f"unknown part {part}")
