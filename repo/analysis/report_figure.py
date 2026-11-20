"""
Summary figure for the report: every estimate of the dipole, as a multiple of
its own sample's kinematic expectation, grouped by the choice that was varied.
Reads only stage results (s13, s14, s15, s17); writes results/report/summary_dipoles.png.

Error bars: the spread of null mocks with the data-driven clustering model
(the model appropriate for configuration A; see AI log entry 45).
"""
import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import config
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def num(stage, part):
    return json.loads((config.RESULTS / stage / f"summary_{part}.json").read_text())["numbers"]


s13 = num("s13_real", "summary")["results"]
s14 = num("s14_ecliptic", "summary")["results"]
s15 = num("s15_quaia", "summary")["results"]
s17 = num("s17_colour", "summary")["results"]["data"]

groups = []
rows = []
for c, lab in (("A", "A: + ecliptic line (S22)"), ("B", "B: dipole only"), ("C", "C: + ecliptic + ℓ = 2, 3"), ("D", "D: + ℓ = 2, 3")):
    r = s13[c]
    rows.append((lab, r["D_over_Dkin"], r["models"]["data"]["D_over_Dkin_err"], r["angle_to_CMB"]))
groups.append(("Fit configuration", rows))
rows = []
for s in ("none", "line", "quad", "cos", "step20", "line+quad", "cov", "line+cov"):
    r = s14[s]["A"]
    dk = r["D"] / r["D_over_Dkin"]
    rows.append((s, r["D_over_Dkin"], r["sigma_D"]["data"] / dk, r["angle_to_CMB"]))
groups.append(("Ecliptic correction shape (config A)", rows))
rows = []
for cut in ("0.80", "0.85", "0.90", "1.00", "1.10"):
    r = s17["A"][cut]
    dk = r["D"] / r["D_over_Dkin"]
    rows.append((f"W1 − W2 ≥ {cut}", r["D_over_Dkin"], r["sigma"] / dk, r["angle_to_CMB"]))
groups.append(("Colour cut (config A)", rows))
rows = []
for s, lab in (("low", "Quaia G < 20.0"), ("high", "Quaia G < 20.5")):
    r = s15[s]["40"]["A"]
    dk = r["D"] / r["D_over_Dkin"]
    rows.append((lab, r["D_over_Dkin"], r["models"]["data"]["sigma_D"] / dk, r["angle_to_CMB"]))
groups.append(("Quaia, |b| ≥ 40° (config A)", rows))

INK, MUTED, MARK = "#1f1f1f", "#6b6b6b", "#2a6fdb"
fig, ax = plt.subplots(figsize=(8.2, 7.6))
y, yt, yl = 0, [], []
for title, rows in groups:
    ax.text(-0.02, y - 0.2, title, transform=ax.get_yaxis_transform(), ha="right", va="center",
            fontsize=10, fontweight="bold", color=INK)
    y -= 1
    for lab, v, e, ang in rows:
        ax.errorbar(v, y, xerr=e, fmt="o", ms=5, color=MARK, ecolor=MARK, elinewidth=1.6, capsize=0)
        ax.text(v + e + 0.08, y, f"{ang:.0f}°", va="center", fontsize=8, color=MUTED)
        yt.append(y); yl.append(lab)
        y -= 1
    y -= 0.6
for x, lab in ((1, "kinematic\nexpectation"), (2, "2 ×")):
    ax.axvline(x, color=MUTED, lw=1, ls=":")
    ax.text(x, 0.6, lab, ha="center", va="bottom", fontsize=8, color=MUTED)
ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=9, color=INK)
ax.set_ylim(y + 0.4, 1.6)
ax.set_xlim(0, 5.2)
ax.set_xlabel("dipole ÷ that sample's kinematic expectation\n(bar: ±1σ with data-driven clustering; grey label: angle from the CMB direction)", fontsize=9)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color="#e6e6e6", lw=0.8)
ax.set_axisbelow(True)
fig.subplots_adjust(left=0.36, right=0.98, top=0.95, bottom=0.1)
out = config.stage_dir("report") / "summary_dipoles.png"
fig.savefig(out, dpi=200)
print("wrote", out)
