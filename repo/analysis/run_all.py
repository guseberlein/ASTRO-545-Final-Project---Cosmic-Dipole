"""
Run every stage in order. See README.md for the running time.

    python run_all.py            # everything
    python run_all.py s05 s06    # only stages whose names start with these
"""
import subprocess, sys, time, pathlib

ROOT = pathlib.Path(__file__).resolve().parent
STEPS = [
    ("s00", ["stages/s00_environment.py"]),
    ("s01", ["stages/s01_footprint.py"]),
    ("s02", ["stages/s02_inputs.py"]),
    ("s03", ["stages/s03_geometry.py"]),
    ("s04", ["stages/s04_mocks.py"]),
    ("s05", ["stages/s05_estimator.py"]),
    ("s06", ["stages/s06_likelihood.py", "closedform"]), ("s06", ["stages/s06_likelihood.py", "gauss_D"]),
    ("s06", ["stages/s06_likelihood.py", "gauss_0"]), ("s06", ["stages/s06_likelihood.py", "mocks_0"]),
    ("s06", ["stages/s06_likelihood.py", "isotropic"]), ("s06", ["stages/s06_likelihood.py", "summary"]),
    ("s07", ["stages/s07_control.py", "mocks", "0"]), ("s07", ["stages/s07_control.py", "mocks", "1"]),
    ("s07", ["stages/s07_control.py", "predict"]), ("s07", ["stages/s07_control.py", "summary"]),
    ("s08", ["stages/s08_main.py", "mocks", "0"]), ("s08", ["stages/s08_main.py", "mocks", "1"]),
    ("s08", ["stages/s08_main.py", "mocks", "2"]), ("s08", ["stages/s08_main.py", "mocks", "3"]),
    ("s08", ["stages/s08_main.py", "predict", "A"]), ("s08", ["stages/s08_main.py", "predict", "B"]),
    ("s08", ["stages/s08_main.py", "predict", "C"]), ("s08", ["stages/s08_main.py", "summary"]),
    ("s09", ["stages/s09_result_plot.py"]),
    ("x01", ["stages/x01_ecliptic_shapes.py"]),
    ("s10", ["stages/s10_dkin.py", "real"]),
    ("s10", ["stages/s10_dkin.py", "boot", "0"]), ("s10", ["stages/s10_dkin.py", "boot", "1"]),
    ("s10", ["stages/s10_dkin.py", "boot", "2"]), ("s10", ["stages/s10_dkin.py", "boot", "3"]),
    ("s10", ["stages/s10_dkin.py", "fake", "0"]), ("s10", ["stages/s10_dkin.py", "fake", "1"]),
    ("s10", ["stages/s10_dkin.py", "fake", "2"]),
    ("s10", ["stages/s10_dkin.py", "fakedip", "0"]), ("s10", ["stages/s10_dkin.py", "fakedip", "1"]),
    ("s10", ["stages/s10_dkin.py", "nvss"]),
    ("s10", ["stages/s10_dkin.py", "summary"]),
    ("s11", ["stages/s11_clustering.py", "pixwin"]),
    ("s11", ["stages/s11_clustering.py", "theory"]),      # needs CAMB; ~5 min and ~2 GB of memory
    ("s11", ["stages/s11_clustering.py", "lognormal"]),
    ("s11", ["stages/s11_clustering.py", "noise"]),
    ("s11", ["stages/s11_clustering.py", "fit"]),
    ("s11", ["stages/s11_clustering.py", "check"]),
    ("s11", ["stages/s11_clustering.py", "summary"]),
    ("s12", ["stages/s12_nullmocks.py", "templates"]),
] + [("s12", ["stages/s12_nullmocks.py", "mocks", m, str(c)]) for m in ("poisson", "lcdm", "data") for c in range(4)] + [
    ("s12", ["stages/s12_nullmocks.py", "summary"]),
    ("s13", ["stages/s13_real.py", "nvss"]), ("s13", ["stages/s13_real.py", "sample"]),
    ("s13", ["stages/s13_real.py", "dipole"]), ("s13", ["stages/s13_real.py", "dkin"]),
] + [("s13", ["stages/s13_real.py", "mocks", m, str(c)]) for m in ("poisson", "lcdm", "data") for c in range(4)] + [
    ("s13", ["stages/s13_real.py", "summary"]),
    ("s13", ["stages/s13_real.py", "leakage"]),
    ("s13", ["stages/s13_real.py", "wobble", "lcdm", "150"]), ("s13", ["stages/s13_real.py", "wobble", "data", "150"]),
    ("s14", ["stages/s14_ecliptic.py", "templates"]), ("s14", ["stages/s14_ecliptic.py", "real"]),
    ("s14", ["stages/s14_ecliptic.py", "matrix"]),
] + [("s14", ["stages/s14_ecliptic.py", "mocks", m, str(c)]) for m in ("lcdm", "data") for c in range(4)] + [
    ("s14", ["stages/s14_ecliptic.py", "summary"]),
    ("s15", ["stages/s15_quaia.py", "maps"]), ("s15", ["stages/s15_quaia.py", "mittal"]),
    ("s15", ["stages/s15_quaia.py", "dkin"]), ("s15", ["stages/s15_quaia.py", "configs"]),
    ("s15", ["stages/s15_quaia.py", "nz"]),
    ("s15", ["stages/s15_quaia.py", "theory"]),           # needs CAMB; a few minutes
    ("s15", ["stages/s15_quaia.py", "clfit", "low"]), ("s15", ["stages/s15_quaia.py", "clfit", "high"]),
] + [("s15", ["stages/s15_quaia.py", "mocks", s, str(cut), m, str(c)])
     for s in ("low", "high") for cut in (30, 40) for m in ("lcdm", "data") for c in range(4)] + [
    ("s15", ["stages/s15_quaia.py", "summary"]),
    ("s15", ["stages/s15_quaia.py", "cuts", "low", "200"]), ("s15", ["stages/s15_quaia.py", "cuts", "high", "200"]),
    ("s16", ["stages/s16_origin.py", "split"]), ("s16", ["stages/s16_origin.py", "real"]),
] + [("s16", ["stages/s16_origin.py", "mocks", h, m, str(c)]) for h in ("kin", "fast", "intr") for m in ("lcdm", "data") for c in range(3)] + [
    ("s16", ["stages/s16_origin.py", "summary"]), ("s16", ["stages/s16_origin.py", "robust"]),
    ("s16", ["stages/s16_origin.py", "composition"]),
    ("s17", ["stages/s17_colour.py", "maps"]), ("s17", ["stages/s17_colour.py", "real"]),
    ("s17", ["stages/s17_colour.py", "safety"]),
] + [("s17", ["stages/s17_colour.py", "mocks", m, str(c)]) for m in ("lcdm", "data") for c in range(5)] + [
    ("s17", ["stages/s17_colour.py", "summary"]),
    ("s18", ["evidence_map.py"]), ("s19", ["report_figure.py"]),
    ("zz", ["check_summary.py"]),
]
only = sys.argv[1:]
for tag, cmd in STEPS:
    if only and not any(tag.startswith(o) for o in only):
        continue
    t0 = time.time()
    print(f"\n=== {' '.join(cmd)}")
    r = subprocess.run([sys.executable] + [str(ROOT / cmd[0])] + cmd[1:], cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"FAILED: {' '.join(cmd)}")
    print(f"    ({time.time() - t0:.0f} s)")
print("\nAll stages finished. See results/ALL_CHECKS.md for every PASS/FAIL.")
