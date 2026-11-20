"""
Stage s10: the kinematic prediction D_kin with an uncertainty. Project Step 3.

Parts (run in this order; each writes summary_<part>.json):
  real        x, alpha and D_kin on the real S22 sample by three methods and a
              grid of fit choices (window, polynomial degree, bin size)
  boot <c>    bootstrap (resampling sources), chunks c = 0..3, 50 each
  fake <c>    fake catalogs with known x and alpha at the cut; every method
              choice is run on them to measure its bias.
              c = 0, 1: counts and mean alpha follow cubic fits to the data;
              c = 2: quadratic fits (a second, smoother realistic shape)
  fakedip <c> end-to-end test of the formula: fake catalogs given random sky
              positions and a boost of 10 beta, cut at W1 < 16.5, dipole fitted
              on the full sky; c = 0, 1, 10 catalogs each
  nvss        preview: x, alpha and D_kin after removing quasars matched to
              NVSS sources (S22 removes NVSS counterparts before measuring x and
              alpha; the full removal, including S22's random homogenizing
              removal, is done in Step 5). Not a check.
  summary     combines the parts, applies the checks, writes the plot

Checks (set before running)
  1. The S22-method default reproduces stage s02 exactly.
  2. The direct-boost values at k = 5, 10, 20 follow D/k = a + b k^2
     (residuals < 0.2% of a).
  3. On the fake catalogs, the default choice of each method recovers the
     true x and alpha with a bias smaller than its statistical error on the
     real data, for both fake families.
  4. End to end: the dipole fitted on boosted fake skies agrees with the
     direct-boost prediction within 3 standard errors.
  5. On the real data, the formula with the local x and alpha agrees with the
     direct boost within 2 bootstrap standard deviations of their difference.

Sample: the S22 footprint (mask s22) with w1cov >= 80, before S22's removal
of NVSS counterparts (that removal is added in Step 5, where D_kin is
recomputed on the final sample). The released catalog continues to
W1 = 16.6, so sources just fainter than the cut are available.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, healpy as hp
import config
from dipole import load_mask, load_inputs, unit_vector, fit_dipole_lsq, NPIX
from dipole.catalog import load_catalog, galactic_pixels
from dipole.measure import alpha_at_limit, x_at_limit, d_kin
from dipole.fluxlimit import (millimag, x_local, alpha_local, d_direct, d_direct_extrapolated,
                              FakeCatalog)
from dipole.checks import StageRecord, read_stage

STAGE = "s10_dkin"
part = sys.argv[1] if len(sys.argv) > 1 else "summary"
chunk = int(sys.argv[2]) if len(sys.argv) > 2 else 0
I = load_inputs()
beta = I["beta"]
M_LIM = 16500

# ---------------------------------------------------------------- method choices
# S22 fit windows: log S < logS_max, i.e. the faintest 0.15, 0.27, 0.52, 0.77, 1.02, 1.27 mag
# of the sample (S22's choice, -1.0, is the faintest 0.27 mag)
SMAX = (-1.05, -1.0, -0.9, -0.8, -0.7, -0.6)
X_S22 = [(smax, deg) for smax in SMAX for deg in (1, 2, 3, 4)]
A_S22 = [(smax, bs) for smax in SMAX for bs in (5000, 10000, 20000)]
X_LOC = [(h, deg, one) for h in (20, 50, 100, 200) for deg in (1, 2) for one in (False, True)]
A_LOC = [(h, one) for h in (20, 50, 100, 200) for one in (False, True)]
DEFAULT = dict(x_s22=(-1.0, 3), a_s22=(-1.0, 10000), x_loc=(50, 2, False), a_loc=(50, False),
               x_loc1=(100, 2, True), a_loc1=(100, True))


def key_x_s22(c): return f"x_s22 logSmax={c[0]} deg={c[1]}"
def key_a_s22(c): return f"a_s22 logSmax={c[0]} bin={c[1]}"
def key_x_loc(c): return f"x_loc half={c[0]}mmag deg={c[1]} {'one' if c[2] else 'two'}-sided"
def key_a_loc(c): return f"a_loc half={c[0]}mmag {'one' if c[1] else 'two'}-sided"


def all_methods(mm, alpha, logS):
    """Every method choice on one catalog (millimag, alpha, log S).
    S22-method fits use only the sample (mm < M_LIM)."""
    s = mm < M_LIM
    ls, al = logS[s], alpha[s]
    out = {}
    for c in X_S22: out[key_x_s22(c)] = x_at_limit(ls, logS_max=c[0], deg=c[1])[0]
    for c in A_S22: out[key_a_s22(c)] = alpha_at_limit(ls, al, logS_max=c[0], binsize=c[1])[0]
    for c in X_LOC: out[key_x_loc(c)] = x_local(mm, M_LIM, half=c[0], deg=c[1], one_sided=c[2])
    for c in A_LOC: out[key_a_loc(c)] = alpha_local(mm, alpha, M_LIM, half=c[0], one_sided=c[1])
    return out


def defaults(mm, alpha, logS, seed):
    """The default choice of each method plus the direct boost (k = 10)."""
    s = mm < M_LIM
    r = dict(x_s22=x_at_limit(logS[s], *DEFAULT["x_s22"])[0],
             a_s22=alpha_at_limit(logS[s], alpha[s], logS_max=DEFAULT["a_s22"][0], binsize=DEFAULT["a_s22"][1])[0],
             x_loc=x_local(mm, M_LIM, *DEFAULT["x_loc"]),
             a_loc=alpha_local(mm, alpha, M_LIM, half=DEFAULT["a_loc"][0], one_sided=DEFAULT["a_loc"][1]),
             x_loc1=x_local(mm, M_LIM, *DEFAULT["x_loc1"]),
             a_loc1=alpha_local(mm, alpha, M_LIM, half=DEFAULT["a_loc1"][0], one_sided=DEFAULT["a_loc1"][1]),
             d_dir10=d_direct(mm, alpha, beta, 10.0, seed=seed))
    r["d_s22"] = d_kin(r["x_s22"], r["a_s22"], beta)
    r["d_loc"] = d_kin(r["x_loc"], r["a_loc"], beta)
    r["d_loc1"] = d_kin(r["x_loc1"], r["a_loc1"], beta)
    return r


DEF_KEYS = ["x_s22", "a_s22", "x_loc", "a_loc", "x_loc1", "a_loc1", "d_dir10", "d_s22", "d_loc", "d_loc1"]


def load_sample():
    """Footprint sources with w1cov >= 80 down to W1 = 16.6."""
    cat = load_catalog(("ra", "dec", "w1", "w1cov", "alpha", "logS"))
    ip = galactic_pixels(cat["ra"], cat["dec"])
    k = (~load_mask("s22")[ip]) & (cat["w1cov"] >= config.W1COV_MIN)
    return millimag(cat["w1"][k]), cat["alpha"][k].astype(float), cat["logS"][k].astype(float)


def logS_from_mm(mm, c0):
    """log S as stored in the catalog: exactly c0 - 0.4 W1, rounded to 1e-4."""
    return np.round(c0 - 0.4 * mm / 1000.0, 4)


# ================================================================= parts
if part == "real":
    rec = StageRecord(STAGE, "real")
    mm, alpha, logS = load_sample()
    s = mm < M_LIM
    rec.number("n_sample_below_cut", int(s.sum()))
    rec.number("n_catalog_to_16.6", int(mm.size))
    X = np.vstack([np.ones(mm.size), mm / 1000.0, alpha]).T
    c, *_ = np.linalg.lstsq(X, logS, rcond=None)
    rec.number("logS_relation", dict(const=c[0], per_mag=c[1], per_alpha=c[2],
                                     rms=float(np.std(logS - X @ c))))
    rec.check("log S depends only on W1 (slope -0.4, no alpha term)",
              abs(c[1] + 0.4) < 1e-6 and abs(c[2]) < 1e-6, f"{c[1]:.7f} per mag, {c[2]:.1e} per unit alpha")
    x0 = x_at_limit(logS[s], -1.0, 3)[0]; a0 = alpha_at_limit(logS[s], alpha[s], -1.0, 10000)[0]
    rec.check("1. S22-method default reproduces stage s02",
              abs(x0 - I["x_ours"]) < 1e-12 and abs(a0 - I["alpha_ours"]) < 1e-12, f"x {x0:.4f}, alpha {a0:.4f}")
    grid = all_methods(mm, alpha, logS)
    rec.number("grid", grid)
    D0, vals, res = d_direct_extrapolated(mm, alpha, beta)
    rec.number("d_direct", dict(limit=D0, k5=vals[0], k10=vals[1], k20=vals[2], residuals=res.tolist()))
    rec.check("2. direct boost follows D/k = a + b k^2", np.max(np.abs(res)) < 0.002 * D0,
              f"D/k = {vals[0]:.6f}, {vals[1]:.6f}, {vals[2]:.6f} -> limit {D0:.6f}; max residual {np.max(np.abs(res)):.1e}")
    rec.number("defaults", defaults(mm, alpha, logS, seed=1))
    rec.write()

elif part == "boot":
    rec = StageRecord(STAGE, f"boot{chunk}")
    mm, alpha, logS = load_sample()
    rows = []
    for i in range(50 * chunk, 50 * (chunk + 1)):
        rng = np.random.default_rng(10_000_000 + i)
        j = rng.integers(0, mm.size, mm.size)
        r = defaults(mm[j], alpha[j], logS[j], seed=10_000_000 + i)
        rows.append([r[k] for k in DEF_KEYS])
    np.save(rec.dir / f"boot_{chunk}.npy", np.array(rows))
    rec.number("n", len(rows))
    rec.write()

elif part == "fake":
    rec = StageRecord(STAGE, f"fake{chunk}")
    mm, alpha, logS = load_sample()
    c0 = read_stage(STAGE, "real")["numbers"]["logS_relation"]["const"]
    deg = 2 if chunk == 2 else 3
    F = FakeCatalog(mm, alpha, deg_n=deg, deg_alpha=deg)
    rec.number("truth", dict(x=F.x_true, alpha=F.alpha_true, D=d_kin(F.x_true, F.alpha_true, beta),
                             degree=deg, p_logn=F.p_logn.tolist(), p_alpha=F.p_alpha.tolist()))
    rows = []
    for i in range(10):
        rng = np.random.default_rng(11_000_000 + 100 * chunk + i)
        m, a = F.draw(rng)
        fm = millimag(m)
        g = all_methods(fm, a, logS_from_mm(fm, c0))
        g["d_direct_limit"] = d_direct_extrapolated(fm, a, beta, seed=i)[0]
        rows.append(g)
    keys = list(rows[0])
    np.save(rec.dir / f"fake_{chunk}.npy", np.array([[r[k] for k in keys] for r in rows]))
    rec.number("keys", keys)
    rec.write()

elif part == "fakedip":
    rec = StageRecord(STAGE, f"fakedip{chunk}")
    mm, alpha, _ = load_sample()
    F = FakeCatalog(mm, alpha)
    K = 10.0
    b = K * beta; gam = 1 / np.sqrt(1 - b * b)
    vhat = unit_vector()
    full = np.zeros(NPIX, bool)
    rows = []
    for i in range(10):
        rng = np.random.default_rng(12_000_000 + 100 * chunk + i)
        m, a = F.draw(rng)
        pred = d_direct(millimag(m), a, beta, K, seed=i) * K           # all orders, same catalog
        n = rng.normal(size=(m.size, 3)); n /= np.linalg.norm(n, axis=1)[:, None]
        mu = n @ vhat                                                  # rest-frame cos(theta)
        perp = n - mu[:, None] * vhat
        mu_o = (mu + b) / (1 + b * mu)                                 # aberration
        sin_r = np.sqrt(np.clip(1 - mu ** 2, 1e-30, None)); sin_o = np.sqrt(np.clip(1 - mu_o ** 2, 0, None))
        n_o = mu_o[:, None] * vhat + perp * (sin_o / sin_r)[:, None]
        delta = gam * (1 + b * mu)                                     # Doppler factor
        m_o = m - 2.5 * (1 + a) * np.log10(delta)
        keep = millimag(m_o) < M_LIM
        cnt = np.bincount(hp.vec2pix(config.NSIDE, *n_o[keep].T), minlength=NPIX).astype(float)
        f = fit_dipole_lsq(cnt, full)
        rows.append([f["amp"], float(f["d"] @ vhat), pred, float(np.sqrt(np.mean(f["sigma"][:3] ** 2)))])
    np.save(rec.dir / f"fakedip_{chunk}.npy", np.array(rows))
    rec.number("truth_linear_D_times_K", d_kin(F.x_true, F.alpha_true, beta) * K)
    rec.write()

elif part == "nvss":
    from astropy.io import fits
    rec = StageRecord(STAGE, "nvss")
    d = fits.open(config.CATALOG)[1].data
    sid = np.array(d["source_id"]).astype(str)
    ip = galactic_pixels(np.array(d["ra"]), np.array(d["dec"]))
    s = (~load_mask("s22")[ip]) & (np.array(d["w1cov"]) >= config.W1COV_MIN) & (np.array(d["w1"]) < config.W1_CUT)
    a, ls = np.array(d["alpha"], float), np.array(d["logS"], float)
    m = fits.open(config.RELEASE / "nvss" / "reference" / "NVSS_CatWISE2020_40arcsec_best_symmetric.fits")[1].data
    msid, S14 = np.array(m["source_id"]).astype(str), np.array(m["S1_4"], float)
    out = {}
    for lab, cut in [("none", None), ("any NVSS match", 0.0), ("NVSS match, S >= 10 mJy", 10.0),
                     ("NVSS match, S >= 20 mJy", 20.0)]:
        k = s.copy()
        if cut is not None:
            k &= ~np.isin(sid, msid[S14 >= cut])
        x = x_at_limit(ls[k], -1.0, 3)[0]; al = alpha_at_limit(ls[k], a[k], -1.0, 10000)[0]
        out[lab] = dict(n=int(k.sum()), removed_percent=100 * (1 - k.sum() / s.sum()), x=x, alpha=al,
                        D=d_kin(x, al, beta))
        print(f"  {lab:26s} removed {out[lab]['removed_percent']:.2f}%  x {x:.4f}  alpha {al:.4f}  D {out[lab]['D']:.5f}")
    rec.number("removal_preview", out)
    rec.note("S22's x = 1.89 is measured after removing NVSS counterparts (their code, wise/code/2_rm_nvss.py). "
             "Removing the 1.0-1.4% of quasars with NVSS matches raises x from 1.878 to 1.889-1.892, which "
             "accounts for the difference between our D_kin (before removal) and S22's.")
    rec.write()

elif part == "summary":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rec = StageRecord(STAGE, "summary")
    R = read_stage(STAGE, "real")["numbers"]
    d = rec.dir
    boot = np.vstack([np.load(d / f"boot_{c}.npy") for c in range(4)])
    B = {k: boot[:, i] for i, k in enumerate(DEF_KEYS)}
    sd = {k: float(np.std(v, ddof=1)) for k, v in B.items()}
    rec.number("n_bootstrap", int(boot.shape[0]))
    rec.number("bootstrap_sd", sd)
    # ---- fakes: bias of every method choice, per curvature family
    fams = {"cubic": [0, 1], "quadratic": [2]}
    bias = {}
    for fam, cs in fams.items():
        keys = read_stage(STAGE, f"fake{cs[0]}")["numbers"]["keys"]
        arr = np.vstack([np.load(d / f"fake_{c}.npy") for c in cs])
        tr = read_stage(STAGE, f"fake{cs[0]}")["numbers"]["truth"]
        bias[fam] = {}
        for j, k in enumerate(keys):
            t = tr["D"] if k == "d_direct_limit" else (tr["x"] if k.startswith("x_") else tr["alpha"])
            v = arr[:, j] - t
            bias[fam][k] = dict(bias=float(v.mean()), se=float(v.std(ddof=1) / np.sqrt(len(v))),
                                truth=t, n=len(v))
    rec.number("fake_bias", bias)
    stat_x = {"x_s22": sd["x_s22"], "x_loc": sd["x_loc"], "x_loc1": sd["x_loc1"]}
    stat_a = {"a_s22": sd["a_s22"], "a_loc": sd["a_loc"], "a_loc1": sd["a_loc1"]}

    def stat_for(k):
        if k.startswith("x_s22"): return sd["x_s22"]
        if k.startswith("a_s22"): return sd["a_s22"]
        if k.startswith("x_loc"): return sd["x_loc1"] if "one-sided" in k else sd["x_loc"]
        if k.startswith("a_loc"): return sd["a_loc1"] if "one-sided" in k else sd["a_loc"]
        return sd["d_dir10"]

    validated = {k: all(abs(bias[f][k]["bias"]) < stat_for(k) for f in fams) for k in bias["cubic"]}
    rec.number("validated_choices", validated)
    dk = [key_x_s22(DEFAULT["x_s22"]), key_a_s22(DEFAULT["a_s22"]), key_x_loc(DEFAULT["x_loc"]),
          key_a_loc(DEFAULT["a_loc"]), key_x_loc(DEFAULT["x_loc1"]), key_a_loc(DEFAULT["a_loc1"]), "d_direct_limit"]
    rec.check("3. default choices recover the true x and alpha on fakes (|bias| < statistical error)",
              all(validated[k] for k in dk),
              "; ".join(f"{k}: {bias['cubic'][k]['bias']:+.4f} / {bias['quadratic'][k]['bias']:+.4f}" for k in dk))
    # ---- what the default choices' biases do to D (recorded, not a check)
    pairs = {"S22 method": (dk[0], dk[1]), "local two-sided": (dk[2], dk[3]), "local one-sided": (dk[4], dk[5])}
    dbias = {}
    for fam, cs in fams.items():
        tr = read_stage(STAGE, f"fake{cs[0]}")["numbers"]["truth"]
        dbias[fam] = {nm: (d_kin(tr["x"] + bias[fam][kx]["bias"], tr["alpha"] + bias[fam][ka]["bias"], beta)
                           / tr["D"] - 1) for nm, (kx, ka) in pairs.items()}
    rec.number("fractional_D_bias_of_default_pairs", dbias)
    rec.note("Check 3 asks for each default's bias to be below its statistical error on the real data. "
             "S22's alpha method (straight line through binned alpha, faintest 0.27 mag) fails it: on fakes "
             "whose mean alpha curves with magnitude as the real data do, it overestimates alpha at the cut by "
             "~0.004-0.005 (3-10 times its statistical error, 0.0014), because the line cannot follow the "
             "steepening of alpha toward the cut. The effect on D_kin is 0.1-0.2% (fractional_D_bias_of_default_pairs). "
             "The criterion was set before running and is kept; the failure is reported, not hidden.")
    # ---- end to end
    fd = np.vstack([np.load(d / f"fakedip_{c}.npy") for c in range(2)])
    ratio = fd[:, 1] / fd[:, 2]
    rm, rse = float(ratio.mean()), float(ratio.std(ddof=1) / np.sqrt(len(ratio)))
    rec.number("fakedip_measured_over_direct", dict(mean=rm, se=rse, n=len(ratio),
                                                    sigma_per_sky=float(np.mean(fd[:, 3]))))
    rec.check("4. boosted fake skies: fitted dipole = direct-boost prediction", abs(rm - 1) < 3 * rse,
              f"{rm:.4f} +/- {rse:.4f}")
    # ---- real data: comparison of methods
    Dd = R["d_direct"]["limit"]
    diff = B["d_loc"] - B["d_dir10"]
    Dl = R["defaults"]["d_loc"]
    rec.check("5. formula with local x, alpha agrees with the direct boost",
              abs(Dl - Dd) < 2 * np.std(diff, ddof=1), f"{Dl:.6f} vs {Dd:.6f}; sd of difference {np.std(diff, ddof=1):.1e}")
    # ---- final number
    G = R["grid"]
    xs = {k: v for k, v in G.items() if k.startswith("x_") and validated[k]}
    als = {k: v for k, v in G.items() if k.startswith("a_") and validated[k]}
    combos = [d_kin(xv, av, beta) for kx, xv in xs.items() for ka, av in als.items()
              if (kx.startswith("x_s22") == ka.startswith("a_s22"))]
    D_lo, D_hi = float(min(combos + [Dd])), float(max(combos + [Dd]))
    meth = 0.5 * (D_hi - D_lo)
    stat = sd["d_dir10"]
    tot = float(np.hypot(stat, meth))
    rec.number("D_kin", dict(value=Dd, stat=stat, method_half_range=meth, total=tot,
                             range_of_validated_choices=[D_lo, D_hi], n_combinations=len(combos)))
    rec.number("D_kin_S22", I["D_kin_from_published"])
    rec.number("x_alpha_defaults", {k: (R["defaults"][k], sd[k]) for k in DEF_KEYS})
    Dobs, eobs = 0.0148, 0.0016
    rr = Dobs / Dd
    rec.number("ratio_S22_Dobs_over_Dkin", dict(value=rr, from_Dobs=rr * eobs / Dobs, from_Dkin=rr * tot / Dd))
    rec.note("Bootstrap resamples sources independently; clustering correlates nearby sources, but x and "
             "alpha are sky averages over 1.6 million sources, so its effect on them is expected to be small.")
    rec.note("Sample is before NVSS-counterpart removal; D_kin is recomputed on the final sample in Step 5.")

    # ---- plot
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
    for a_, pre, lab, tkey in [(ax[0], "x_", "x at the cut", "x"), (ax[1], "a_", "alpha at the cut", "alpha")]:
        ks = [k for k in G if k.startswith(pre)]
        y = np.arange(len(ks))
        v = np.array([G[k] for k in ks]); e = np.array([stat_for(k) for k in ks])
        col = ["#2a78d6" if validated[k] else "#b0aea8" for k in ks]
        a_.errorbar(v, y, xerr=e, fmt="none", ecolor=col, lw=1.2)
        a_.scatter(v, y, c=col, s=14, zorder=3)
        a_.set_yticks(y); a_.set_yticklabels(ks, fontsize=6)
        ref = config.X_S22 if tkey == "x" else config.ALPHA_S22
        a_.axvline(ref, color="k", ls=":", lw=1); a_.set_xlabel(lab + "  (dotted: S22; grey: failed fake test)")
        a_.invert_yaxis(); a_.grid(axis="x", alpha=0.3)
    names = ["S22 method\n(default)", "local,\ntwo-sided", "local,\none-sided", "direct\nboost", "S22\npublished"]
    vals = [R["defaults"]["d_s22"], R["defaults"]["d_loc"], R["defaults"]["d_loc1"], Dd, I["D_kin_from_published"]]
    errs = [sd["d_s22"], sd["d_loc"], sd["d_loc1"], stat, 0]
    ax[2].errorbar(range(5), np.array(vals) * 1e3, yerr=np.array(errs) * 1e3, fmt="o", color="#2a78d6", capsize=3)
    ax[2].axhspan((Dd - tot) * 1e3, (Dd + tot) * 1e3, color="#eb6834", alpha=0.15, label="adopted ± total")
    ax[2].set_xticks(range(5)); ax[2].set_xticklabels(names, fontsize=8)
    ax[2].set_ylabel("D_kin  (× 10⁻³)"); ax[2].legend(frameon=False, loc="lower left"); ax[2].grid(axis="y", alpha=0.3)
    ax[2].set_ylim(min(vals) * 1e3 - 0.06, max(vals) * 1e3 + 0.04)
    ax[2].set_title(f"D_kin = {Dd*1e3:.3f} ± {tot*1e3:.3f} × 10⁻³", loc="left", fontsize=11)
    plt.tight_layout(); plt.savefig(d / "dkin.png", dpi=130); plt.close()
    rec.write()
else:
    sys.exit(f"unknown part {part}")
