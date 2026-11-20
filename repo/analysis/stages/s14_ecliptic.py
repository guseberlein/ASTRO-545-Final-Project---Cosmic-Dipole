"""
Stage s14: does the shape of the ecliptic correction matter? Project Step 6
(the student's idea).

The prototype and S22 correct the WISE scanning pattern with a straight line
in |ecliptic latitude|. This stage asks four questions:
  1. Which correction shapes describe the real sky better than the straight
     line, by more than chance would allow?
  2. What does each correction do to the real dipole (configs A and C)?
  3. Does any correction remove the excess large-scale power at l = 2-4
     found in Step 4 (which would show that power is a survey artifact)?
  4. If the true pattern had shape X at its measured size, how much would
     each correction bias the dipole (planted x fitted matrix), and do the
     real data's shifts between corrections match any X?

Shapes (dipole.templates): line, quad, cos, step20 (|elat| > 20 deg), cov
(mean W1 coverage per pixel to the power -1/2: photometric noise scales as
coverage^-1/2), and combinations line+quad, line+cov, and "NS line"
(separate straight lines north and south of the ecliptic). All multiply the
density.

Parts:
  templates          coverage map; template maps; estimator checks
  real               every correction on the real final sample (Step 5's)
  matrix             noiseless planted x fitted, shapes at their measured size
  mocks <m> <c>      null mocks (straight-line pattern, kinematic dipole,
                     clustering model m = lcdm or data from Step 4, sub-pixel
                     scatter, Poisson), every correction fitted; chunks c = 0..3
                     of 50. They calibrate how much fit improvement, dipole
                     shift and residual power chance alone produces.
  summary            checks, results, figure

Checks (set before running)
  1. With the line template alone, the general estimator equals Step 4/5's
     fit_multipoles_mult on the real map (configs A and C) to 1e-10.
  2. Noiseless: a map with dipole x (line + quad + cov pattern) is recovered
     exactly (dipole and all three template amplitudes) by the matching fit.
  3. Noiseless matrix diagonal: every planted shape fitted with its own
     correction returns the planted dipole exactly (configs A and C).
  4. Null mocks, line correction, config A: unbiased (mean vector = D_kin
     toward the CMB, within 3 standard errors) for both clustering models.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, healpy as hp
from astropy.io import fits
import config
from dipole import (load_mask, load_inputs, unit_vector, cos_theta_map, ecliptic_latitude_map,
                    fit_multipoles_mult, NPIX)
from dipole.catalog import galactic_pixels, count_map
from dipole.clustering import gaussian_cls_for_lognormal, lognormal_delta, ylm_templates
from dipole.templates import fit_general_mult, deviance, template_maps
from dipole.checks import StageRecord, read_stage

STAGE = "s14_ecliptic"
part = sys.argv[1] if len(sys.argv) > 1 else "summary"
OUT = config.stage_dir(STAGE)
LMAX = 3 * config.NSIDE - 1
mask = load_mask("s22"); seen = ~mask
elat = ecliptic_latitude_map(); ae = np.abs(elat)
dhat = unit_vector()
Y, _ = ylm_templates((2, 3))
RBANDS = [(2, 2), (3, 3), (4, 4), (5, 5), (6, 10)]
CFG = {"A": None, "C": Y}


def load_templates():
    z = np.load(OUT / "templates.npz")
    single, sets = template_maps(elat, z["w1cov_pix"])
    return single, sets


def fit_set(cnt, single, names, cfg):
    return fit_general_mult(cnt, mask, [single[n] for n in names], CFG[cfg])


def residual_bands(cnt, model):
    r = np.zeros(NPIX); r[seen] = cnt[seen] / model[seen] - 1
    cl = hp.anafast(r, lmax=LMAX) / seen.mean()
    return np.array([cl[a:b + 1].mean() for a, b in RBANDS])


def angle(a, b):
    return float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))


def run_all_sets(cnt, single, sets):
    """Every correction, configs A and C: dipole vectors, deviances, and
    (config A) residual low-l power."""
    D = np.zeros((len(sets), 2, 3)); V = np.zeros((len(sets), 2)); B = np.zeros((len(sets), len(RBANDS)))
    S = {}
    for i, (nm, names) in enumerate(sets.items()):
        for j, cfg in enumerate(("A", "C")):
            f = fit_set(cnt, single, names, cfg)
            D[i, j] = f["d"]; V[i, j] = deviance(cnt, f["model"], mask)
            if cfg == "A":
                B[i] = residual_bands(cnt, f["model"])
                S[nm] = f["s"].tolist()
    return D, V, B, S


# ================================================================= parts
if part == "templates":
    rec = StageRecord(STAGE, "templates")
    c = fits.open(config.CATALOG)[1].data
    ip = galactic_pixels(np.array(c["ra"]), np.array(c["dec"]))
    n_all = np.bincount(ip, minlength=NPIX)
    w1cov_pix = np.bincount(ip, np.array(c["w1cov"], float), NPIX) / np.maximum(n_all, 1)
    np.savez(OUT / "templates.npz", w1cov_pix=w1cov_pix)
    single, sets = template_maps(elat, w1cov_pix)
    rec.check("template maps finite on the footprint", all(np.all(np.isfinite(t[seen])) for t in single.values()))
    rec.number("w1cov_footprint_percentiles_5_50_95", np.percentile(w1cov_pix[seen], [5, 50, 95]).tolist())
    rec.number("corr_cov_template_with_abs_elat", float(np.corrcoef(single["cov"][seen], ae[seen])[0, 1]))
    cnt = np.load(config.RESULTS / "s13_real" / "final_counts.npy")
    errs = []
    for cfg in ("A", "C"):
        g = fit_set(cnt, single, ["line"], cfg)
        r = fit_multipoles_mult(cnt, mask, ae, CFG[cfg])
        errs.append(float(np.max(np.abs(g["d"] - r["d"]))))
    rec.check("1. general estimator = fit_multipoles_mult with the line template (A, C)", max(errs) < 1e-10,
              f"max differences {errs[0]:.1e}, {errs[1]:.1e}")
    I = load_inputs(); D = I["D_kin_from_published"]
    s_true = np.array([-9e-4, 2e-6, 0.5])
    pat = 1 + s_true[0] * single["line"] + s_true[1] * single["quad"] + s_true[2] * (single["cov"] - single["cov"][seen].mean())
    m = 70 * pat * (1 + D * cos_theta_map(dhat)); m[mask] = 0
    f = fit_general_mult(m, mask, [single["line"], single["quad"], single["cov"] - single["cov"][seen].mean()])
    e_d = float(np.linalg.norm(f["d"] - D * dhat)); e_s = float(np.max(np.abs(f["s"] / s_true - 1)))
    rec.check("2. noiseless multi-template map recovered exactly (dipole and amplitudes)", e_d < 1e-10 and e_s < 1e-8,
              f"dipole error {e_d:.1e}; largest relative amplitude error {e_s:.1e}")
    rec.write()

elif part == "real":
    rec = StageRecord(STAGE, "real")
    single, sets = load_templates()
    cnt = np.load(config.RESULTS / "s13_real" / "final_counts.npy")
    D, V, B, S = run_all_sets(cnt, single, sets)
    np.savez(OUT / "real.npz", D=D, V=V, B=B)
    rec.number("sets", list(sets))
    rec.number("template_amplitudes_A", S)
    rec.number("dipoles", {nm: {cfg: dict(D=float(np.linalg.norm(D[i, j])), l=hp.vec2ang(D[i, j], lonlat=True)[0][0],
                                         b=hp.vec2ang(D[i, j], lonlat=True)[1][0], angle_to_CMB=angle(D[i, j], dhat))
                                for j, cfg in enumerate(("A", "C"))} for i, nm in enumerate(sets)})
    rec.write()

elif part == "matrix":
    rec = StageRecord(STAGE, "matrix")
    single, sets = load_templates()
    S = read_stage(STAGE, "real")["numbers"]["template_amplitudes_A"]
    Dk = read_stage("s13_real", "dkin")["numbers"]["dkin"]["D_direct"]
    planted = [k for k in sets if k != "none"]
    res, diag_err = {}, []
    for X in planted:
        pat = 1 + sum(s * single[n] for s, n in zip(S[X], sets[X]))
        m = 70 * pat * (1 + Dk * cos_theta_map(dhat)); m[mask] = 0
        res[X] = {}
        for nm, names in sets.items():
            res[X][nm] = {}
            for cfg in ("A", "C"):
                f = fit_set(m, single, names, cfg)
                res[X][nm][cfg] = dict(d=f["d"].tolist(), amp_over_Dkin=f["amp"] / Dk, angle=angle(f["d"], dhat))
                if nm == X:
                    diag_err.append(float(np.linalg.norm(f["d"] - Dk * dhat)))
    rec.number("planted_x_fitted", res)
    rec.check("3. noiseless diagonal: each shape fitted with its own correction is exact (A and C)",
              max(diag_err) < 1e-10, f"largest error {max(diag_err):.1e}")
    rec.write()

elif part == "mocks":
    model, chunk = sys.argv[2], int(sys.argv[3])
    rec = StageRecord(STAGE, f"mocks_{model}_{chunk}")
    single, sets = load_templates()
    tr = read_stage("s13_real", "dipole")["numbers"]["final_trend"]
    Dk = read_stage("s13_real", "dkin")["numbers"]["dkin"]["D_direct"]
    from dipole.mocks import expected_counts
    lam = expected_counts(cos_theta_map(dhat), ae, mask, N0=tr["intercept"], s=tr["frac_slope"], D=Dk)
    cl = np.load(config.RESULTS / "s11_clustering" / f"cl_{model}.npy")[: LMAX + 1]
    w = np.load(config.RESULTS / "s11_clustering" / "pixwin.npy")[: LMAX + 1]
    clg = gaussian_cls_for_lognormal(cl * w ** 2, LMAX)
    pixvar = read_stage("s11_clustering", "fit")["numbers"]["subpixel_pixel_variance"]
    n = 50
    Ds, Vs, Bs = [], [], []
    for j in range(n):
        i = chunk * n + j
        rf = np.random.default_rng(24_000_000 + i)
        mu = lam * (1 + lognormal_delta(clg, LMAX, rf)) * rf.gamma(1 / pixvar, pixvar, NPIX)
        cm = np.random.default_rng(25_000_000 + i).poisson(mu).astype(float)
        D_, V_, B_, _ = run_all_sets(cm, single, sets)
        Ds.append(D_); Vs.append(V_); Bs.append(B_)
    np.savez(OUT / f"mocks_{model}_{chunk}.npz", D=np.array(Ds), V=np.array(Vs), B=np.array(Bs))
    rec.number("n", n)
    rec.write()

elif part == "summary":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rec = StageRecord(STAGE, "summary")
    sets = read_stage(STAGE, "real")["numbers"]["sets"]
    iL = sets.index("line")
    R = np.load(OUT / "real.npz")
    Dk = read_stage("s13_real", "dkin")["numbers"]["dkin"]["D_direct"]
    M = {}
    for m in ("lcdm", "data"):
        z = [np.load(OUT / f"mocks_{m}_{c}.npz") for c in range(4)]
        M[m] = {k: np.concatenate([q[k] for q in z]) for k in ("D", "V", "B")}
    ok = True; zs = {}
    for m in M:
        d = M[m]["D"][:, iL, 0, :]
        z = (d.mean(0) - Dk * dhat) / (d.std(0, ddof=1) / np.sqrt(len(d)))
        zs[m] = z.tolist(); ok &= bool(np.all(np.abs(z) < 3))
    rec.check("4. null mocks, line correction, config A: unbiased", ok, "; ".join(f"{m}: z = {np.round(v, 2).tolist()}" for m, v in zs.items()))
    out = {}
    for i, nm in enumerate(sets):
        out[nm] = {}
        # 1. fit improvement over the straight line, against chance
        dv = float(R["V"][iL, 0] - R["V"][i, 0])
        out[nm]["deviance_improvement_over_line_A"] = dv
        out[nm]["p_chance"] = {m: float(np.mean((M[m]["V"][:, iL, 0] - M[m]["V"][:, i, 0]) >= dv)) for m in M}
        # 2. dipole and its shift from the line correction, against chance
        for j, cfg in enumerate(("A", "C")):
            d = R["D"][i, j]; sh = d - R["D"][iL, j]
            msh = {m: M[m]["D"][:, i, j, :] - M[m]["D"][:, iL, j, :] for m in M}
            # mock shifts: mean (the systematic shift this correction causes when the truth is the line) and
            # scatter about that mean (chance); the real shift is compared with both
            out[nm][cfg] = dict(D=float(np.linalg.norm(d)), D_over_Dkin=float(np.linalg.norm(d) / Dk),
                                angle_to_CMB=angle(d, dhat), shift_from_line=float(np.linalg.norm(sh)),
                                shift_vec=sh.tolist(),
                                shift_angle=angle(d, R["D"][iL, j]) if nm != "line" else 0.0,
                                mock_mean_shift={m: v.mean(0).tolist() for m, v in msh.items()},
                                mock_shift_scatter_rms={m: float(np.sqrt(np.sum(np.var(v, axis=0, ddof=1)))) for m, v in msh.items()},
                                real_minus_mock_mean_shift={m: float(np.linalg.norm(sh - v.mean(0))) for m, v in msh.items()},
                                sigma_D={m: float(np.sqrt(np.mean(np.var(M[m]["D"][:, i, j, :], axis=0, ddof=1)))) for m in M})
        # 3. residual low-l power against a fixed baseline: lcdm mocks (true pattern = line) fitted with the
        #    RIGHT correction (line). First version compared each correction with mocks fitted by the same
        #    correction; for corrections that do not contain the line, the mocks' own misfit raised that
        #    baseline and made the data look consistent for the wrong reason (kept as *_same_fit_baseline).
        mb, sb = M["lcdm"]["B"][:, iL, :].mean(0), M["lcdm"]["B"][:, iL, :].std(0, ddof=1)
        out[nm]["residual_power_z_vs_lcdm"] = dict(zip([f"{a}-{b}" for a, b in RBANDS], ((R["B"][i] - mb) / sb).tolist()))
        out[nm]["residual_power_data_over_lcdm_mean"] = dict(zip([f"{a}-{b}" for a, b in RBANDS], (R["B"][i] / mb).tolist()))
        mb2, sb2 = M["lcdm"]["B"][:, i, :].mean(0), M["lcdm"]["B"][:, i, :].std(0, ddof=1)
        out[nm]["residual_power_z_same_fit_baseline"] = dict(zip([f"{a}-{b}" for a, b in RBANDS], ((R["B"][i] - mb2) / sb2).tolist()))
    rec.number("results", out)
    rec.note("The comparison of real shifts with shifts predicted from planted shapes is circular when the planted "
             "size is fitted to the same real sky: in config A the predicted line -> line+cov shift equals the real "
             "one by construction (2.89 vs 2.87 x 1e-3, 0 deg apart). The independent evidence is the mock "
             "calibration: chance shift scatter and chance fit improvement on skies whose true pattern is the line.")
    # significance of the real dipole with the line and line+cov corrections (Gaussian fitted to null mocks)
    from scipy.stats import norm
    sig = {}
    g = np.random.default_rng(14)
    for nm in ("line", "line+cov"):
        i = sets.index(nm); sig[nm] = {}
        for j, cfg in enumerate(("A", "C")):
            d = R["D"][i, j]; Dobs = float(np.linalg.norm(d)); sig[nm][cfg] = {}
            for m in M:
                x = M[m]["D"][:, i, j, :]; mu = x.mean(0); Lc = np.linalg.cholesky(np.cov(x.T))
                hits = sum(int(np.sum(np.linalg.norm(mu + g.standard_normal((2_000_000, 3)) @ Lc.T, axis=1) >= Dobs))
                           for _ in range(10))
                pa = max(hits, 0.5) / 2e7
                sig[nm][cfg][m] = dict(D=Dobs, D_over_Dkin=Dobs / Dk, p=pa, sigma=float(norm.isf(pa)), hits=hits,
                                       n_mocks=len(x))
    rec.number("significance_line_vs_linecov", sig)
    rec.note("Significance from 200 null mocks per model (Gaussian fitted to their dipole vectors); Step 5 used 1000. "
             "The data-driven clustering model was fitted to residuals after the LINE correction, so for line+cov in "
             "config A it includes octupole power that the coverage term removes; in config C the octupole is fitted, "
             "so this does not matter there.")
    # figure
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(sets))
    for j, (cfg, c) in enumerate((("A", "#2a78d6"), ("C", "#eb6834"))):
        amps = [out[nm][cfg]["D"] * 1e3 if nm != "NS line" else np.nan for nm in sets]
        errs = [out[nm][cfg]["sigma_D"]["data"] * 1e3 if nm != "NS line" else np.nan for nm in sets]
        ax[0].errorbar(x + (j - 0.5) * 0.25, amps, yerr=errs, fmt="o", color=c, capsize=3,
                       label=f"config {cfg}" + (" (+ ℓ = 2, 3 fitted)" if cfg == "C" else ""))
    ax[0].axhline(Dk * 1e3, color="k", ls=":", lw=1); ax[0].text(-0.45, Dk * 1e3 + 0.4, "D_kin", va="bottom")
    ax[0].set_ylim(0, 32)
    iN = sets.index("NS line")
    ax[0].text(iN, 21, "off scale:\nA %.0f ± %.0f\nC %.0f ± %.0f\n(dipole\nundetermined)" % (
        out["NS line"]["A"]["D"] * 1e3, out["NS line"]["A"]["sigma_D"]["data"] * 1e3,
        out["NS line"]["C"]["D"] * 1e3, out["NS line"]["C"]["sigma_D"]["data"] * 1e3), ha="center", va="top", fontsize=8)
    ax[0].set_xticks(x); ax[0].set_xticklabels(sets, rotation=30, ha="right")
    ax[0].set_ylabel("real dipole amplitude (× 10⁻³)\nerror bars: data-driven null spread")
    ax[0].legend(frameon=False); ax[0].grid(axis="y", alpha=0.3)
    ax[0].set_title("Real CatWISE dipole with each ecliptic correction", loc="left", fontsize=11)
    TOP = 10
    for k, (bnd, c) in enumerate(zip(RBANDS[:3], ("#0b0b0b", "#7a7975", "#b0aea8"))):
        vals = np.array([out[nm]["residual_power_z_vs_lcdm"][f"{bnd[0]}-{bnd[1]}"] for nm in sets])
        ax[1].bar(x + (k - 1) * 0.27, np.minimum(vals, TOP), 0.25, color=c, label=f"ℓ = {bnd[0]}")
        for xi, v in zip(x + (k - 1) * 0.27, vals):
            if v > TOP:
                ax[1].text(xi, TOP + 0.2, f"{v:.0f}", ha="center", va="bottom", fontsize=7, rotation=90)
    ax[1].set_ylim(-3, TOP + 2.5)
    ax[1].axhline(0, color="k", lw=0.8); ax[1].axhspan(-2, 2, color="#2a78d6", alpha=0.08)
    ax[1].set_xticks(x); ax[1].set_xticklabels(sets, rotation=30, ha="right")
    ax[1].set_ylabel("leftover power above ΛCDM\n(standard deviations; band: ±2)")
    ax[1].legend(frameon=False); ax[1].grid(axis="y", alpha=0.3)
    ax[1].set_title("Does any correction remove the excess large-scale power?", loc="left", fontsize=11)
    plt.tight_layout(); plt.savefig(OUT / "ecliptic_shapes.png", dpi=130); plt.close()
    rec.write()
else:
    sys.exit(f"unknown part {part}")
