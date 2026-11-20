"""Recording checks and key numbers for each stage (results/<stage>/summary.json)."""
import time
import config
from .io import write_json, read_json


class StageRecord:
    """Collects PASS/FAIL checks and the stage's key numbers, prints them as
    they are added, and writes results/<stage>/summary.json.

    For stages run in parts, each part writes its own file
    (summary_<part>.json) so that parts can be rerun independently."""

    def __init__(self, stage, part=None):
        self.stage, self.part = stage, part
        self.dir = config.stage_dir(stage)
        self.checks, self.numbers, self.notes = [], {}, []

    def check(self, name, passed, detail=""):
        passed = bool(passed)
        self.checks.append(dict(name=name, passed=passed, detail=detail))
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
        return passed

    def number(self, key, value):
        self.numbers[key] = value
        return value

    def note(self, text):
        self.notes.append(text)
        print("  note:", text)

    def write(self):
        name = "summary.json" if self.part is None else f"summary_{self.part}.json"
        out = dict(stage=self.stage, part=self.part,
                   written=time.strftime("%Y-%m-%d %H:%M:%S"),
                   all_checks_passed=all(c["passed"] for c in self.checks),
                   checks=self.checks, numbers=self.numbers, notes=self.notes)
        write_json(out, self.dir / name)
        n_fail = sum(not c["passed"] for c in self.checks)
        print(f"  -> {self.dir.name}/{name}: {len(self.checks)} checks, {n_fail} failed")
        return out


def read_stage(stage, part=None):
    name = "summary.json" if part is None else f"summary_{part}.json"
    return read_json(config.RESULTS / stage / name)
