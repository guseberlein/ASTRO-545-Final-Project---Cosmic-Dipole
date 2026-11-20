# Can estimator choices make or hide the cosmic dipole excess?

Code for the capstone project of Gus Eberlein (Graduate Cosmology, November 2026): a reanalysis of the CatWISE2020 and Quaia quasar number-count dipoles. The report (submitted separately as a PDF) describes the analysis and its results; its Appendix A repeats these instructions.

## Requirements

Python 3.10 or newer:

```
pip install -r analysis/requirements.txt
```

## Data

Download two public releases and place them in this folder (or anywhere, and set the environment variable `DIPOLE_DATA` to that folder):

- the Secrest et al. (2022) release, Zenodo record 6784602: unpack `secrest+22_accepted.tgz` to get `secrest+22_accepted/secrest+22_accepted/`;
- the Quaia catalogues, Zenodo record 10403370: put `quaia_G20.0.fits`, `quaia_G20.5.fits`, the two `selection_function_NSIDE64_*.fits` files and the unzipped `selection_function_template_maps/` folder in `quaia/`.

`analysis/README.md` lists exactly which files are read.

## Running

```
cd analysis
python run_all.py
```

This runs every stage in order (about an hour on a two-core laptop) and writes everything to `analysis/results/`:

- `results/ALL_CHECKS.md`: every pre-set check, PASS or FAIL, with each failure explained;
- `results/<stage>/summary_<part>.json`: each stage's checks and key numbers;
- `results/<stage>/*.png`: the figures;
- `../evidence_map.md`: every number quoted in the report's claims, regenerated from the results and checked against the claim text.

All random numbers come from fixed seeds, so a rerun reproduces every number in the report. `analysis/README.md` describes each stage, how to run a single stage or part, and the method choices.
