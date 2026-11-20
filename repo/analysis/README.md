# Cosmic dipole analysis: code

This folder contains all the code for the project. It measures the number-count dipole of the CatWISE quasar sample and tests how analysis choices affect the measured dipole, using mock skies in which the true answer is known.

The research question: *Can choices in how the dipole is estimated on the CatWISE footprint (the mask, the ecliptic-latitude correction, and whether higher multipoles are fitted) produce or hide the factor of two to three between the measured quasar dipole and the kinematic prediction; and if the excess survives in CatWISE2020 and Quaia, does it depend on the sample's spectral index, as an intrinsic rather than kinematic origin would?*

Every stage runs a set of **checks** and records whether each one passed. All the checks are collected in `results/ALL_CHECKS.md`.

---

## 1. What you need

**Software.** Python 3.10 or newer:

```
pip install -r requirements.txt     # numpy scipy healpy astropy matplotlib camb
```

The results were produced with Python 3.10.12, numpy 2.2.6, healpy 1.20.1, scipy 1.15.3, astropy 6.1.7 and matplotlib 3.10.9. Stage s09 showed the mock results are identical under numpy 2.5. A from-scratch rerun in an empty folder reproduced the stored results exactly (see the report, Appendix A). CAMB is needed only by stage s11, part `theory`, which needs about 2 GB of memory.

**Data.** Two public releases, arranged like this:

```
<data folder>/
    secrest+22_accepted/
        secrest+22_accepted/    <- S22 release (Zenodo 6784602), unpacked from secrest+22_accepted.tgz
    quaia/                      <- Quaia (Zenodo 10403370)
        quaia_G20.0.fits
        quaia_G20.5.fits
        selection_function_NSIDE64_G20.0.fits
        selection_function_NSIDE64_G20.5.fits
        selection_function_template_maps/selection_function_template_maps/map_unwisescan_NSIDE64.npy
                                   (from selection_function_template_maps.zip)
```

From the S22 release the code reads seven files:
`wise/reference/catwise_agns.fits`, `wise/reference/exclude_master_revised.fits`,
`secrest+21/secrest+21/secrest+21_nside64.fits`, `nvss/reference/NVSS.fit`,
`nvss/reference/NVSS_CatWISE2020_40arcsec_best_symmetric.fits`, `nvss/reference/nvss_artifacts.fits`
and `reference/haslam408_dsds_Remazeilles2014_512.fits`.

By default the data folder is the folder that contains this code folder. To keep the data elsewhere, set the environment variable `DIPOLE_DATA` to the data folder. Results are always written to `results/` inside this folder.

## 2. How to run

From inside this folder:

```
python run_all.py              # every stage in order (about an hour on a two-core laptop)
python run_all.py s13 s14      # only the stages whose names start with these
```

Each stage can also be run on its own. The longer stages run in parts, so that each part finishes in a few minutes. The part names are listed at the top of each stage's file:

```
python stages/s13_real.py mocks lcdm 0      # ... chunks 0-3
python stages/s13_real.py summary
```

**Order matters.** s01 makes the masks, and s02 measures the input numbers (`results/s02_inputs/inputs.json`). Every later stage reads both, and most stages read earlier stages' results.

**To see the results:**
- `results/ALL_CHECKS.md` lists every check with PASS or FAIL, and explains each failure. Regenerate it with `python check_summary.py`.
- `results/<stage>/summary_<part>.json` holds each part's checks and key numbers.
- `results/<stage>/*.png` holds each stage's figures; `report_figure.py` makes the report's summary figure.
- `python evidence_map.py` regenerates every number in the report's claims, checks it against the claim text, and writes `../evidence_map.md`.

All random numbers come from fixed seeds, so a rerun reproduces every number.

## 3. Layout

| Path | Contents |
|---|---|
| `config.py` | file locations, constants (CMB dipole, sample cuts), published S22 values |
| `dipole/geometry.py` | pixel vectors, angle to the CMB direction, ecliptic latitude |
| `dipole/catalog.py` | reading the catalog, the S22 selection, count maps |
| `dipole/masks.py` | the three footprints (S22 rebuilt from its recipe) |
| `dipole/measure.py` | ecliptic-trend fit; x and α at the flux limit (S22's method); D_kin |
| `dipole/clustering.py` | redshift distribution, CAMB and exact spectra, pixel window, lognormal skies, spectrum measurement, ℓ = 2, 3 templates |
| `dipole/templates.py` | dipole fit with any number of multiplicative correction shapes; Poisson deviance; the candidate shapes |
| `dipole/quaia.py` | Quaia: equatorial-grid geometry, Gaia spectral indices, dipole fits with the selection function as a fixed factor |
| `dipole/fluxlimit.py` | x and α from a narrow window around the cut; D_kin by Doppler-boosting every source; fake catalogs with known x and α |
| `dipole/mocks.py` | Poisson mock skies |
| `dipole/estimators.py` | the dipole estimators (see section 5) |
| `dipole/likelihood.py` | likelihood of the dipole amplitude A, L(A) |
| `dipole/checks.py`, `dipole/io.py` | recording checks; reading and writing files |
| `stages/` | one script per stage (section 4) |
| `run_all.py`, `check_summary.py`, `evidence_map.py`, `report_figure.py` | running, collecting checks, the evidence map and the summary figure |

## 4. Stages and their checks

| Stage | What it does | Checks |
|---|---|---|
| s00_environment | packages and data present | N_side = 64 gives 49,152 pixels; the input files exist |
| s01_footprint | fallback, S21 and S22 masks | fallback sky fraction = 0.500; S21 map total = published 1,355,352; S22 rebuild: 51% of the sky masked and 1.6 million quasars, as S22 states |
| s02_inputs | density, ecliptic trend, x, α, D_kin | the method reproduces S21's published trend; x = 1.878 and α = 1.061 against S22's 1.89 and 1.06; D_kin = 0.00727 |
| s03_geometry | geometry maps | unit vectors; mean of cos θ = 0; poles and CMB pixel |
| s04_mocks | mock generator | pixel means and Poisson variances; total-count mean and spread |
| s05_estimator | least-squares estimator | exact recovery of noiseless maps; error formula; agreement with healpy; **reproduces S21's published dipole from the real S21 map**; explanation of the additive-template bias |
| s06_likelihood | L(A) | closed-form test; grid convergence; the error model matches Poisson mocks; bias of the amplitude summaries |
| s07_control | 500 mocks, dipole only | unbiased vector; honest error bars; amplitude matches its predicted bias |
| s08_main | 500 mocks, dipole and ecliptic pattern; fits A, B, C | each fit behaves as its noiseless result predicts; **result: leaving out the template inflates the dipole by +9.06% ± 0.20% of D_inj** |
| s09_result_plot | result figure | numbers agree with s08; reproducible from the stored seeds |
| x01_ecliptic_shapes | exploration: other pattern shapes; real data with a coverage template | none yet (exploratory) |
| s10_dkin | D_kin with an uncertainty: three methods (S22's fits, a narrow window around the cut, a direct Doppler boost), a grid of fit choices, 200 bootstrap resamples, fake catalogs with known x and α, boosted fake skies | S22's method reproduces s02; the boost behaves as expected with speed; the default choices recover the truth on fakes (**fails for S22's α fit: biased by +0.004–0.005, i.e. 0.1–0.2% in D_kin; see the stage notes**); the fitted dipole on boosted fake skies equals the prediction; the formula and the direct boost agree. **Result: D_kin = (7.219 ± 0.032) × 10⁻³ before NVSS removal** |

| s11_clustering | clustering models for the mocks: HEALPix pixel window; CAMB spectrum (Planck 2018, N(z) read from S21 Fig. 3, Tiwari et al.'s bias) with an exact low-ℓ cross-check; lognormal generator; two models fitted to the data's measured spectrum by forward modelling (ΛCDM shape with one amplitude; data-driven, with free low-ℓ power) | N(z) mean redshift; CAMB vs exact; generator tests; each model reproduces the data's spectrum. **Five of seven checks fail; each is explained in the stage notes** (tolerances at the 3% level near ℓ = 2 and ℓ = 191; data below the noise level at ℓ = 5–7; the first models lacked sub-pixel clustering and were revised) |
| s12_nullmocks | 1000 null mocks per clustering model (none, ΛCDM, data-driven), each fitted in four configurations (A: + ecliptic; B: dipole only; C: + ecliptic + ℓ = 2, 3; D: + ℓ = 2, 3) | noiseless recovery in every configuration; Poisson mocks reproduce the prototype exactly; A and C unbiased with clustering. **Result: the null spread of the dipole is 1.5–3.4 × 10⁻³ per component depending on the clustering model and the configuration** |

| s13_real | the real CatWISE dipole: S22's NVSS sample and final quasar sample rebuilt from their code; an exact replica of S22's estimator; our four configurations; D_kin on the final sample; null mocks for the final sample; significance | NVSS sample 508,156 vs S22's 508,144; NVSS-counterpart fraction (**fails: 1.3495% against S22's rounded 1.4%, 8 quasars below the rounding range; the tolerance was too tight, and the result agrees**); the replica reproduces S22's dipole (0.0152 toward (239.5°, 30.2°)); healpy and Poisson maximum likelihood agree with our fit; D_kin by formula = direct boost; null mocks unbiased. **Result: D = 0.0154 (2.1 × D_kin) with the ecliptic template, 0.0204 (2.8 × D_kin) when ℓ = 2, 3 are also fitted; significance 2.6–5.5σ depending on the clustering model and configuration** |

| s14_ecliptic | the shape of the ecliptic correction (the student's idea): eight shapes (line, quadratic, cosine, step, coverage, and combinations, including separate north and south slopes) on the real sky; 400 null mocks to calibrate chance; planted × fitted matrix | the general estimator reproduces Step 5's; noiseless multi-shape maps recovered exactly; matrix diagonal exact; null mocks unbiased. **Result: line + coverage fits better than the line by more than chance (p ≤ 0.01), removes the octupole excess, and lowers the dipole by 5% (A) / 3% (C); the excess survives (2.0–2.7 × D_kin). Separate north and south slopes are degenerate with the dipole** |

| s15_quaia | the Quaia dipole (G < 20.0 and G < 20.5): equatorial-grid analysis (the selection function is equatorial); Mittal et al.'s model at Galactic cuts 0–40°; D_kin from Gaia BP − RP spectral indices; configurations A, C and A + unWISE scan; clustering models and null mocks at 30° and 40° | selection function's frame confirmed; Mittal et al.'s amplitudes (**fails at small cuts; matches at 30–40° within 2 × 10⁻³; a different pixel weighting reproduces them**); Poisson ML = least squares; D_kin formula = direct boost; null mocks unbiased. **Result: Quaia's dipole falls with the Galactic cut; at 40° (G < 20.0) it is 1.6 × D_kin, 18° from the CMB direction, consistent with both the kinematic value and a factor-two excess. A cut scan (30–60°) shows the map is clean from ~35–40° (G < 20.0) and ~40–45° (G < 20.5); with ℓ = 2, 3 fitted the dipole is unmeasurable (errors 10–29 × 10⁻³); at clean cuts the two samples lean weakly in opposite directions, so Quaia cannot resolve the question** |

| s16_origin | the origin test: the final CatWISE sample split into three groups by spectral index; each group's Doppler factor K = 2 + x(1 + α); group dipoles fitted as d = K b + a (b: scales like motion; a: common to all groups); mocks with one shared clustering field under three hypotheses (CMB motion only, faster motion, common excess) | groups add up to the full sample; K by direct boost = formula; on kinematic-only mocks b is unbiased and the test is calibrated (false-alarm rate 4–5%). **Result: K varies only 5.6–6.4 between groups, so the planned test has almost no power (3–7% detection on mocks). But the three group dipoles differ far beyond what any of the hypotheses allows (0.016, 0.019, 0.024; the high-α group points 56° from the CMB, at ecliptic latitude 6°): p ≈ 10⁻⁵–10⁻¹³, still ≤ 2 × 10⁻⁴ if each group has its own independent clustering** |
| s17_colour | colour tests: (A) the full sample rebuilt with W1 − W2 ≥ 0.80, 0.85, 0.90, 1.00, 1.10 and refitted; (B) a colour-composition template (smoothed fraction of low-α quasars) added to the fit; mocks with one clustered sky thinned to each cut, so the change between cuts is compared with what clustering and Poisson noise allow | samples are nested; on noiseless kinematic skies every fit gives zero change between cuts; mock changes centred on zero. **Result: without the template the dipole grows from 2.1 to 3.8 × D_kin and turns from 26° to 63° off the CMB as the cut tightens (change significant from 0.85 on, p ≤ 0.01); with the template the change is not significant up to 0.90 but is from 1.00 on (p ≤ 0.02). At the standard 0.80 cut the template barely moves the dipole (2.13 → 2.14 × D_kin) and its amplitude is not significant (z ≈ 0.8)** |
| s18_evidence (`evidence_map.py`) | the evidence map (project Step 9): ten central claims, each number read from a stage's results file; writes `../evidence_map.md` | every number quoted in a claim equals the value regenerated from the results files (73/73 at the time of writing). s16 also gained a part `composition` (each α group's share against WISE coverage, and the dipole of each share map), which recomputes Step 8 diagnosis numbers first obtained in throwaway commands (AI log entry 59) |

The literature check (project Step 2) is in `../literature_check.md` (not part of the code package).

## 5. Choices that differ from the standard method

- **Requirements for s11.** The theory part needs CAMB (`pip install camb`) and about 2 GB of memory for 5 minutes; CAMB's default accuracy is 10–20% off at ℓ < 40 for this redshift window, so the stage raises it. healpy's pixel-window table must be downloaded, which was not possible here, so the window is computed directly.

- **Multiplicative ecliptic template.** Gibelyou & Huterer (2012) add templates to the density. The scanning effect multiplies the density, and with an additive template a noiseless mock returns 1.0155 × D_inj. `fit_dipole_lsq_mult` fits n/n̄ = a(1 + s t)(1 + d·n̂) and recovers the mock exactly. The additive version (`fit_dipole_lsq`) is kept for comparison.
- **Amplitude bias.** The peak of L(A) is biased high by about 1–2% at our signal-to-noise, because the errors differ between directions. With equal errors in every direction it is unbiased. The checks therefore test the dipole *vector*, which is unbiased, and test the amplitude against its predicted bias.
- **NVSS counterparts.** S22 removes the quasars that also appear in its NVSS radio sample. Stages s02–s12 use the sample before this removal; s13 rebuilds S22's removal exactly and recomputes D_kin and the null mocks on the final sample. This matters for D_kin: removing them raises x from 1.878 to about 1.89 (s10, part `nvss`), so D_kin is recomputed on the final sample.
- **Magnitudes as integers.** W1 is stored as float32 with 0.001-mag resolution. Binning the float values directly puts some sources on the wrong side of bin edges and makes histograms lumpy by ~5%, so `dipole/fluxlimit.py` works in integer millimagnitudes.

## 6. Relation to other folders

- `../prototype/`: the week-11 prototype, kept unchanged as a record (not part of the code package). This code reproduced it exactly (`verify_against_prototype.py`, run on the project computer: 64 of 64 items identical).
- `../AI_log.md`: every place where the AI tools were wrong, incomplete or overconfident, and how each was caught.
