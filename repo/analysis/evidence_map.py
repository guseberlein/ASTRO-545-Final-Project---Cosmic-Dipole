"""
Evidence map (project Step 9).

Every central claim of the project is listed with
  - the numbers that support it, each read from a results file written by a stage,
  - the check(s) that make the numbers trustworthy,
  - the AI-log entries where the claim was challenged or corrected.

Each number appears in the claim text as it will be quoted in the report
("stated"). This script regenerates every number from the results files and
checks that, rounded the same way, it equals the stated value. If a stage is
rerun and a number moves, this check fails and the claim text must be updated.

Writes ../evidence_map.md and results/s18_evidence/summary.json.
Run:  python evidence_map.py      (after the stages it reads; see run_all.py)
"""
import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import config
from dipole.checks import StageRecord

R = config.RESULTS


def get(stage_part, path):
    """stage_part 'stage/part' (results/<stage>/summary_<part>.json); path keys separated by '/'."""
    stage, part = stage_part.split("/")
    obj = json.loads((R / stage / f"summary_{part}.json").read_text())["numbers"]
    for k in path.split("/"):
        obj = obj[int(k)] if isinstance(obj, list) else obj[k]
    return obj


def num(src, path, fmt, stated, scale=1.0):
    return dict(src=src, path=path, fmt=fmt, stated=stated, value=lambda: get(src, path) * scale)


def derived(label, fn, fmt, stated, sources):
    return dict(src=", ".join(sources), path=label, fmt=fmt, stated=stated, value=fn)


# ---------------------------------------------------------------- derived quantities
S14_SETS = ["none", "line", "quad", "cos", "step20", "line+quad", "cov", "line+cov"]   # NS line is unconstrained


def s14_range(key, f, cfg="A"):
    return lambda: f(get("s14_ecliptic/summary", f"results/{s}/{cfg}/{key}") for s in S14_SETS)


def sig_range(f):
    vals = []
    for c in "ABCD":
        for m in ("poisson", "lcdm", "data"):
            vals.append(get("s13_real/summary", f"results/{c}/models/{m}/sigma_amplitude"))
    for s in ("line", "line+cov"):
        for c in "AC":
            for m in ("lcdm", "data"):
                vals.append(get("s14_ecliptic/summary", f"significance_line_vs_linecov/{s}/{c}/{m}/sigma"))
    return f(vals)


def power(f):
    v = []
    for m in ("lcdm", "data"):
        for c in "AC":
            v.append(get("s16_origin/summary", f"results/{m}/{c}/calibration_R1/fast/frac_reject_bcmb"))
            v.append(get("s16_origin/summary", f"results/{m}/{c}/calibration_R1/intr/frac_reject_a0"))
    return 100 * f(v)


FITS16 = ["line (A)", "line+cov", "cov", "line + l=2,3 (C)", "line+cov + l=2,3"]


def worst_p():
    return max(get("s16_origin/robust", f"equal_dipole_tests/{m}/{k}/p_independent_worstcase")
               for m in ("lcdm", "data") for k in FITS16)


# ---------------------------------------------------------------- the claims
CLAIMS = [
 dict(id="Q1-a", part="Can estimator choices produce or hide the excess?",
      claim="The kinematic expectation is known to better than 1%. On the final CatWISE sample, "
            "D_kin = {0} × 10⁻³ from a direct Doppler boost of every source and {1} × 10⁻³ from the formula "
            "with locally measured x = {2} and α = {3}. S22's polynomial method gives {4} × 10⁻³.",
      nums=[num("s13_real/dkin", "dkin/D_direct", "{:.3f}", "7.265", 1e3),
            num("s13_real/dkin", "dkin/D_local", "{:.3f}", "7.265", 1e3),
            num("s13_real/dkin", "dkin/x_local", "{:.3f}", "1.889"),
            num("s13_real/dkin", "dkin/alpha_local", "{:.3f}", "1.059"),
            num("s13_real/dkin", "dkin/D_s22method", "{:.3f}", "7.284", 1e3)],
      check="Step 3 (s10): three methods with no shared fitting choices agree within 0.3%; fake catalogs with known "
            "x and α are recovered; the dipole fitted on boosted fake skies agrees with the prediction (1.012 ± 0.005; see AI log entry 61).",
      log="Entries 31–35 (bias of S22's polynomial fits; float32 binning; stale files)."),

 dict(id="Q1-b", part="Can estimator choices produce or hide the excess?",
      claim="Our pipeline reproduces the published CatWISE result. With S22's choices (mask, NVSS removal, linear "
            "ecliptic correction; configuration A) the dipole is D = {0} × 10⁻³ toward (l, b) = ({1}°, {2}°), "
            "{3} × D_kin and {4}° from the CMB direction (S22: 14.8 ± 1.6 × 10⁻³ toward (238°, 31°)). "
            "The sample has {5} quasars.",
      nums=[num("s13_real/summary", "results/A/D", "{:.1f}", "15.4", 1e3),
            num("s13_real/summary", "results/A/l", "{:.0f}", "240"),
            num("s13_real/summary", "results/A/b", "{:.0f}", "30"),
            num("s13_real/summary", "results/A/D_over_Dkin", "{:.2f}", "2.13"),
            num("s13_real/summary", "results/A/angle_to_CMB", "{:.0f}", "26"),
            num("s13_real/sample", "n_final", "{:,.0f}", "1,578,750")],
      check="Step 5 (s13): rebuilt NVSS sample 508,156 vs S22's 508,144; healpy's dipole fit agrees to 10⁻¹⁷; a Poisson "
            "maximum-likelihood fit agrees within 0.04σ; the Poisson null spread (1.6 × 10⁻³) equals S22's quoted error.",
      log="Entry 43 (replica 0.0152 vs 0.0148)."),

 dict(id="Q1-c", part="Can estimator choices produce or hide the excess?",
      claim="The shape of the ecliptic correction neither creates nor hides the excess. Across eight correction "
            "shapes that the data constrain (none, line, quadratic, cosine, step, line + quadratic, coverage, "
            "line + coverage), the configuration-A dipole stays between {0} and {1} × D_kin and between {2}° and "
            "{3}° from the CMB direction; with ℓ = 2, 3 also fitted (configuration C) it stays between {5} and {6} "
            "× D_kin. Allowing different north and south slopes leaves the dipole unconstrained "
            "(σ_D = {4} × 10⁻³, larger than the dipole). The mask itself was not varied: S22's mask is used "
            "throughout, and its effect enters through the multipole leakage in Q1-d.",
      nums=[derived("min D/D_kin over 8 shapes", s14_range("D_over_Dkin", min), "{:.2f}", "2.00", ["s14_ecliptic/summary"]),
            derived("max D/D_kin over 8 shapes", s14_range("D_over_Dkin", max), "{:.2f}", "2.24", ["s14_ecliptic/summary"]),
            derived("min angle over 8 shapes", s14_range("angle_to_CMB", min), "{:.0f}", "26", ["s14_ecliptic/summary"]),
            derived("max angle over 8 shapes", s14_range("angle_to_CMB", max), "{:.0f}", "33", ["s14_ecliptic/summary"]),
            num("s14_ecliptic/summary", "results/NS line/A/sigma_D/lcdm", "{:.1f}", "10.1", 1e3),
            derived("min D/D_kin over 8 shapes, config C", s14_range("D_over_Dkin", min, "C"), "{:.2f}", "2.72", ["s14_ecliptic/summary"]),
            derived("max D/D_kin over 8 shapes, config C", s14_range("D_over_Dkin", max, "C"), "{:.2f}", "2.86", ["s14_ecliptic/summary"])],
      check="Step 6 (s14): the general estimator reproduces Step 5's to 10⁻¹³ and recovers noiseless multi-shape skies "
            "exactly; the real dipole's shift between corrections matches the shift predicted by mocks.",
      log="Entries 47–49 (moving residual-power baseline; circular shift test; the 24° coverage shift overinterpreted)."),

 dict(id="Q1-d", part="Can estimator choices produce or hide the excess?",
      claim="Fitting higher multipoles raises the excess and doubles its uncertainty. On the |b| < 30° mask the "
            "octupole (ℓ = 3) leaks into the dipole at {0}% of its own size (ℓ = 2: {1}%; ℓ = 5: {2}%). Fitting "
            "ℓ = 2, 3 alongside the dipole (configuration C) gives {3} × D_kin, {4}° from the CMB direction, with "
            "σ_D = {5} × 10⁻³ (ΛCDM clustering) instead of {6} × 10⁻³.",
      nums=[num("s13_real/leakage", "leak_into_dipole_per_unit_pattern_rms/3/rms_fraction", "{:.0f}", "119", 100),
            num("s13_real/leakage", "leak_into_dipole_per_unit_pattern_rms/2/rms_fraction", "{:.0f}", "3", 100),
            num("s13_real/leakage", "leak_into_dipole_per_unit_pattern_rms/5/rms_fraction", "{:.0f}", "25", 100),
            num("s13_real/summary", "results/C/D_over_Dkin", "{:.2f}", "2.81"),
            num("s13_real/summary", "results/C/angle_to_CMB", "{:.0f}", "35"),
            num("s13_real/summary", "results/C/models/lcdm/sigma_D", "{:.1f}", "3.8", 1e3),
            num("s13_real/summary", "results/A/models/lcdm/sigma_D", "{:.1f}", "1.9", 1e3)],
      check="Leakage computed from noiseless single-multipole skies through the same estimator (s13 leakage); "
            "with clustering off, Poisson mocks reproduce the prototype to 10⁻¹⁵.",
      log="Entries 44 (the octupole, not the quadrupole, leaks) and 46 (configuration C still depends on the clustering model)."),

 dict(id="Q1-e", part="Can estimator choices produce or hide the excess?",
      claim="The significance depends more on the assumed clustering than on the estimator. Configuration A gives "
            "{0}σ (Poisson only), {1}σ (ΛCDM clustering) and {2}σ (clustering measured from the data); "
            "configuration C gives {3}σ, {4}σ and {5}σ. Over all fit configurations, ecliptic corrections and "
            "clustering models, the significance ranges from {6}σ to {7}σ. No choice we tested removes the excess.",
      nums=[num("s13_real/summary", "results/A/models/poisson/sigma_amplitude", "{:.1f}", "5.5"),
            num("s13_real/summary", "results/A/models/lcdm/sigma_amplitude", "{:.1f}", "4.3"),
            num("s13_real/summary", "results/A/models/data/sigma_amplitude", "{:.1f}", "2.6"),
            num("s13_real/summary", "results/C/models/poisson/sigma_amplitude", "{:.1f}", "4.3"),
            num("s13_real/summary", "results/C/models/lcdm/sigma_amplitude", "{:.1f}", "3.2"),
            num("s13_real/summary", "results/C/models/data/sigma_amplitude", "{:.1f}", "3.7"),
            derived("min significance", lambda: sig_range(min), "{:.1f}", "2.2", ["s13_real/summary", "s14_ecliptic/summary"]),
            derived("max significance", lambda: sig_range(max), "{:.1f}", "5.5", ["s13_real/summary", "s14_ecliptic/summary"])],
      check="Step 4 (s12): without clustering a Gaussian fitted to null mocks gives 4.83σ for S22's amplitude, matching "
            "Bashir et al.'s replica (4.82σ) and S21 (4.9σ). Null mocks are unbiased in the configurations that model the "
            "ecliptic pattern. Values above ~5.5σ are limited by the number of mocks (no mock reached the real value).",
      log="Entries 45 (data-driven spread is appropriate for A, not one-sided inflation), 46, 36–41 (CAMB accuracy, "
          "sub-pixel clustering, pixel window)."),

 dict(id="Q1-f", part="Can estimator choices produce or hide the excess?",
      claim="The colour cut is a further estimator choice, and it moves the excess. Tightening W1 − W2 ≥ 0.80 to "
            "≥ 1.10 grows the configuration-A dipole from {0} to {1} × D_kin and turns it from {2}° to {3}° off the "
            "CMB direction; the change is far beyond what clustering and noise allow (p = {4}). A template for the "
            "sky-varying colour mix barely changes the standard sample ({5} × D_kin; template amplitude {6}σ from "
            "zero) and does not remove the change at the tightest cut (p = {7}).",
      nums=[num("s17_colour/summary", "results/lcdm/A/0.80/D_over_Dkin", "{:.2f}", "2.13"),
            num("s17_colour/summary", "results/lcdm/A/1.10/D_over_Dkin", "{:.2f}", "3.81"),
            num("s17_colour/summary", "results/lcdm/A/0.80/angle_to_CMB", "{:.0f}", "26"),
            num("s17_colour/summary", "results/lcdm/A/1.10/angle_to_CMB", "{:.0f}", "63"),
            num("s17_colour/summary", "results/lcdm/A/1.10/p_change_from_0p80", "{:.0e}", "3e-15"),
            num("s17_colour/summary", "results/lcdm/A+frac/0.80/D_over_Dkin", "{:.2f}", "2.14"),
            num("s17_colour/summary", "results/lcdm/colour_template_amplitude_0p80/A+frac/z", "{:.1f}", "0.9"),
            num("s17_colour/summary", "results/lcdm/A+frac/1.10/p_change_from_0p80", "{:.0e}", "3e-07")],
      check="Colour tests (s17): samples nested; on noiseless kinematic skies every fit gives zero change between cuts; "
            "mock changes centred on zero.",
      log="Entry 58 (the scan cannot say whether noise at the cut or a real difference between quasars is responsible)."),

 dict(id="Q2-a", part="Does the excess survive in Quaia?",
      claim="Quaia can neither confirm nor rule out the excess. At the Galactic cut where the map becomes clean "
            "(40°), the G < 20.0 sample gives {0} × D_kin, {1}° from the CMB direction: a purely kinematic sky "
            "would give a dipole this large with p = {2}, and a sky with twice the kinematic dipole would give one "
            "this small with p = {3}. The G < 20.5 sample gives {4} × D_kin, {5}° from the CMB (p = {6} and {7}). "
            "With ℓ = 2, 3 fitted, σ_D = {8} × 10⁻³, larger than Quaia's D_kin of {9} × 10⁻³.",
      nums=[num("s15_quaia/cuts_low", "by_cut/40/A/D_over_Dkin", "{:.2f}", "1.59"),
            num("s15_quaia/cuts_low", "by_cut/40/A/angle_to_CMB", "{:.0f}", "18"),
            num("s15_quaia/cuts_low", "by_cut/40/A/p_as_high_if_1x", "{:.2f}", "0.33"),
            num("s15_quaia/cuts_low", "by_cut/40/A/p_as_low_if_2x", "{:.2f}", "0.11"),
            num("s15_quaia/cuts_high", "by_cut/40/A/D_over_Dkin", "{:.2f}", "2.45"),
            num("s15_quaia/cuts_high", "by_cut/40/A/angle_to_CMB", "{:.0f}", "47"),
            num("s15_quaia/cuts_high", "by_cut/40/A/p_as_high_if_1x", "{:.3f}", "0.015"),
            num("s15_quaia/cuts_high", "by_cut/40/A/p_as_low_if_2x", "{:.2f}", "0.74"),
            num("s15_quaia/cuts_low", "by_cut/40/C/sigma_D", "{:.0f}", "10", 1e3),
            num("s15_quaia/dkin", "dkin/low/D_direct", "{:.1f}", "5.3", 1e3)],
      check="Step 7 (s15): the selection function is equatorial (correlation 0.83–0.89 on the equatorial grid, −0.03 on "
            "the Galactic grid); with Mittal et al.'s colour our D_kin equals their corrected 0.0048; formula and direct "
            "boost agree within 0.4%; Mittal et al.'s amplitudes are matched at 30–40° but not at small cuts.",
      log="Entries 50 (Mittal reproduction failure: pixel weighting), 52–55 (ML divergence, missing √a, "
          "'adds nothing', '40° is cleanest' asserted)."),

 dict(id="Q2-b", part="Does the excess depend on spectral index?",
      claim="The planned spectral-index test has no power on CatWISE. Splitting by α changes the Doppler factor "
            "K = 2 + x(1 + α) only from {0} to {1}, so on mocks where the excess is real (kinematic or intrinsic) "
            "the test detects it in only {2}–{3}% of skies, barely above the 5% false-alarm rate.",
      nums=[num("s16_origin/summary", "results/lcdm/A/R1.0/group_K/2", "{:.1f}", "5.6"),
            num("s16_origin/summary", "results/lcdm/A/R1.0/group_K/0", "{:.1f}", "6.4"),
            derived("min detection rate", lambda: power(min), "{:.0f}", "3", ["s16_origin/summary"]),
            derived("max detection rate", lambda: power(max), "{:.0f}", "7", ["s16_origin/summary"])],
      check="Step 8 (s16): the groups add up to the full sample; each group's K by direct boost equals the formula "
            "within 0.3%; on kinematic-only mocks the motion term is unbiased and the test fires 4–5% of the time.",
      log="Entries 56 (the planned test assumed a lever arm it did not have) and 57 (α noise bracketed, not measured)."),

 dict(id="Q2-c", part="Does the excess depend on spectral index?",
      claim="Instead, quasars of different colour have different dipoles. The three α groups have D = {0}, {1} "
            "and {2} × 10⁻³; the high-α group points {3}° from the CMB direction. Equal dipoles are rejected at "
            "p = {4} (shared clustering, configuration A) and at p ≤ {5} even in the worst case where each group "
            "has its own independent clustering, for every ecliptic correction tried.",
      nums=[num("s16_origin/robust", "equal_dipole_tests/lcdm/line (A)/groups/0/D", "{:.1f}", "16.0", 1e3),
            num("s16_origin/robust", "equal_dipole_tests/lcdm/line (A)/groups/1/D", "{:.1f}", "19.4", 1e3),
            num("s16_origin/robust", "equal_dipole_tests/lcdm/line (A)/groups/2/D", "{:.1f}", "23.9", 1e3),
            num("s16_origin/robust", "equal_dipole_tests/lcdm/line (A)/groups/2/angle_to_CMB", "{:.0f}", "56"),
            num("s16_origin/robust", "equal_dipole_tests/lcdm/line (A)/p_shared", "{:.0e}", "3e-12"),
            derived("largest worst-case p over 5 fits × 2 models", worst_p, "{:.0e}", "2e-04", ["s16_origin/robust"])],
      check="As Q2-b; the worst case adds the data-driven clustering variance of every group independently.",
      log="Entry 57."),

 dict(id="Q2-d", part="Does the excess depend on spectral index?",
      claim="The colour mix changes across the sky with WISE's scan depth. From the least- to the most-scanned fifth "
            "of the sky, the low-α share falls from {0}% to {1}% (slope {2}σ) and the high-α share rises from {3}% to "
            "{4}% (slope {5}σ). The high-α share map has a dipole of {6}% toward ({7}°, {8}°), close to the direction "
            "of the high-minus-low group dipole difference ({9}% toward ({10}°, {11}°)). So the α split is a test of "
            "colour-dependent selection, not of the excess's origin.",
      nums=[num("s16_origin/composition", "composition/groups/0/share_by_w1cov_bin/0", "{:.1f}", "25.4", 100),
            num("s16_origin/composition", "composition/groups/0/share_by_w1cov_bin/4", "{:.1f}", "23.8", 100),
            num("s16_origin/composition", "composition/groups/0/slope_z", "{:.0f}", "-15"),
            num("s16_origin/composition", "composition/groups/2/share_by_w1cov_bin/0", "{:.1f}", "42.1", 100),
            num("s16_origin/composition", "composition/groups/2/share_by_w1cov_bin/4", "{:.1f}", "44.3", 100),
            num("s16_origin/composition", "composition/groups/2/slope_z", "{:.0f}", "17"),
            num("s16_origin/composition", "composition/groups/2/share_dipole/amp", "{:.1f}", "1.6", 100),
            num("s16_origin/composition", "composition/groups/2/share_dipole/l", "{:.0f}", "153"),
            num("s16_origin/composition", "composition/groups/2/share_dipole/b", "{:.0f}", "12"),
            num("s16_origin/composition", "composition/high_minus_low_dipole_A/amp", "{:.1f}", "2.4", 100),
            num("s16_origin/composition", "composition/high_minus_low_dipole_A/l", "{:.0f}", "154"),
            num("s16_origin/composition", "composition/high_minus_low_dipole_A/b", "{:.0f}", "18")],
      check="Binomial errors only (clustering of the colour mix ignored), so the σ values are indicative. The low-α "
            "group carries almost the whole ecliptic trend (slope −0.0019/deg against −0.0001/deg for high α), as S22's "
            "colour-scatter mechanism predicts.",
      log="Entry 59 (these numbers were first computed in throwaway commands and quoted as 1.7% and 5–18σ; recomputed "
          "reproducibly as 1.6% and 4–17σ)."),
]


def render(c):
    vals, rows, ok = [], [], True
    for n in c["nums"]:
        v = n["value"]()
        s = n["fmt"].format(v)
        good = (s == n["stated"])
        ok &= good
        vals.append(n["stated"])
        rows.append((n["src"], n["path"], s, n["stated"], good))
    return c["claim"].format(*vals), rows, ok


if __name__ == "__main__":
    rec = StageRecord("s18_evidence", "summary")
    md = ["# Evidence map", "",
          "*Generated by `analysis/evidence_map.py` from the stage results; do not edit by hand. Every number in a "
          "claim is regenerated from the file and key shown, and the check at the bottom confirms that each one, "
          "rounded as quoted, equals the value in the claim.*", ""]
    n_all, n_bad, current = 0, 0, None
    for c in CLAIMS:
        text, rows, ok = render(c)
        if c["part"] != current:
            current = c["part"]
            md += [f"## {current}", ""]
        md += [f"### {c['id']}", "", text, "",
               "| Number | Regenerated from (results file : key) | Matches |", "|---|---|---|"]
        for src, path, s, st, good in rows:
            md.append(f"| {st} | `{src}` : `{path}` | {'yes' if good else f'**NO (code gives {s})**'} |")
        md += ["", f"**Check:** {c['check']}", "", f"**Challenged or corrected:** {c['log']}", ""]
        n_all += len(rows); n_bad += sum(not r[4] for r in rows)
        rec.number(c["id"], {r[1]: r[2] for r in rows})
    rec.check("every number in the claim table equals the value regenerated from the results files",
              n_bad == 0, f"{n_all - n_bad}/{n_all} match")
    md += ["---", f"{n_all - n_bad} of {n_all} numbers match their regenerated values."]
    (config.ROOT.parent / "evidence_map.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    rec.write()
    for c in CLAIMS:
        for src, path, s, st, good in render(c)[1]:
            if not good:
                print(f"  MISMATCH {c['id']}: {src} {path}: code {s}, stated {st}")
