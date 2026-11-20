"""Reading and writing the files that stages pass to each other."""
import json
import numpy as np
import healpy as hp

import config


def save_mask(mask, name):
    """Write a mask (True = masked) as results/s01_footprint/mask_<name>.fits."""
    path = config.stage_dir("s01_footprint") / f"mask_{name}.fits"
    hp.write_map(str(path), np.asarray(mask).astype(np.int16), overwrite=True,
                 coord="G", column_names=["MASKED"], dtype=np.int16)
    return path


def load_mask(name="s22"):
    """Footprint mask built by stage s01; True = masked."""
    path = config.RESULTS / "s01_footprint" / f"mask_{name}.fits"
    return hp.read_map(str(path), dtype=np.int16).astype(bool)


def load_inputs():
    """Input numbers measured by stage s02 (results/s02_inputs/inputs.json)."""
    return read_json(config.RESULTS / "s02_inputs" / "inputs.json")


def _clean(v):
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray):
        return _clean(v.tolist())
    return v


def write_json(obj, path):
    with open(path, "w") as f:
        json.dump(_clean(obj), f, indent=2)


def read_json(path):
    with open(path) as f:
        return json.load(f)
