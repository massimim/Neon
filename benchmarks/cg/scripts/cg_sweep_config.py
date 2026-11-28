#!/usr/bin/env python3
"""Parse and validate the CG sweep YAML (see ``cg-sweep.yml``).

The CG counterpart of ``benchmarks/lbm_ldc/scripts/lbm_sweep_config.py``: the same layout
(global keys, top-level matrix defaults, ``experiments: name -> {type, <matrix keys>}``),
value checks and decomposition variants (``pencil-direct`` / ``brick-direct``), with the
keys ``cg_poisson`` takes. There are no configs, precisions or transports: single-process
grids (``dense``, ``disg``) run in one process on G GPUs, the distributed ones under
``mpiexec -n G`` with NCCL.

Reports are ``<data_dir>/json/cg-<label>-n<N>-g<G>.json``:

- ``runs(exp)`` enumerates the (grid, decomposition, hp_halo, overlap, gpus) cells of one
  experiment, ``label(...)`` names their reports and ``all_runs(cfg)`` merges every
  experiment into the deduplicated list the run script executes.
- Run as a script it reads the YAML text from ``$CG_SWEEP_CONFIG`` (experiments from
  ``$CG_SWEEP_ONLY``, comma or space separated, empty = all), lets the env var of a global
  key override it, and prints ``VAR=value`` lines plus ``EXPERIMENTS=`` and ``RUNS=``.

Values end up in shell variables, docker / mpiexec arguments and file names, so only
plain tokens (letters, digits, ``. _ : / -``) are accepted, and experiment names must be
single path components.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lbm_ldc" / "scripts"))
import lbm_sweep_config as lsc  # noqa: E402

SweepConfigError = lsc.SweepConfigError
DISTRIBUTED_GRIDS = lsc.DISTRIBUTED_GRIDS
DIRECT_SUFFIX = lsc.DIRECT_SUFFIX
split_decomposition = lsc.split_decomposition
decomposition_variants = lsc.decomposition_variants
gpus_of = lsc.gpus_of
overlap_tag = lsc.overlap_tag
split_names = lsc.split_names

SERIES = "cg"
# Global keys: (env var of cg-sweep-run.sh, kind, default)
GLOBAL_KEYS: dict[str, tuple[str, str, object]] = {
    "runner": ("RUNNER", "word", "docker"),
    "warmup": ("WARMUP", "int", 100),
    "iters_per_rep": ("ITERS_PER_REP", "int", 200),
    "reps": ("REPS", "int", 5),
    "overlap": ("OVERLAP", "word", "standard"),
    "reference_label": ("REFERENCE_LABEL", "name", "dense"),
    "data_dir": ("DATA_DIR", "path", "benchmarks/cg/data"),
    "sm_arch": ("SM_ARCH", "arch", "auto"),
    "image": ("IMAGE", "word", "carbon-cuda-nccl-pinned:local"),
    "build_dir": ("BUILD_DIR", "path", "build-mpi"),
    "skip_build": ("SKIP_BUILD", "bool", False),
    "force": ("FORCE", "bool", False),
    "run_timeout": ("RUN_TIMEOUT", "int", 900),
}
# Matrix keys: (kind, default); allowed at the top level (defaults) and per experiment.
MATRIX_KEYS: dict[str, tuple[str, object]] = {
    "sizes": ("ints", list(range(256, 513, 64))),
    "gpus": ("ints", [1, 2, 4, 8]),
    "grids": ("words", ["dense_distributed"]),
    "decompositions": ("words", ["slab"]),
    "gpus_pencil": ("ints", [4, 8]),
    "gpus_brick": ("ints", [8]),
    "hp_halo": ("words", ["off"]),
    "halo_exchange": ("words", ["staged"]),
    "overlaps": ("words", None),
    "plots": ("words", None),
}
# Experiment type -> the figures plot_cg_scaling.py can draw for it (all by default).
TYPES: dict[str, list[str]] = {
    "strong_scaling": ["vs_gpus", "direct_speedup"],
}


def _parse_matrix(where: str, raw: dict) -> dict[str, object]:
    out = {}
    for key, value in raw.items():
        if key not in MATRIX_KEYS:
            raise SweepConfigError(f"{where}: unknown key {key!r} (matrix keys: "
                                   f"{', '.join(MATRIX_KEYS)})")
        out[key] = lsc._parse_value(key, MATRIX_KEYS[key][0], value)
    return out


def load_sweep_config(text: str, env: dict[str, str] | None = None,
                      only: list[str] | None = None) -> dict[str, object]:
    """Validate YAML text; returns the global keys plus ``experiments``.

    ``env`` (e.g. ``os.environ``) overrides global keys: a key whose env var
    (``GLOBAL_KEYS``) is set takes that value, in the run script's format (0/1 bools).
    ``only`` keeps the named experiments (all when empty).
    """
    try:
        import yaml
    except ImportError:
        raise SweepConfigError("PyYAML is not installed: pip install pyyaml "
                               "(or apt install python3-yaml)") from None

    raw = yaml.safe_load(text) or {}
    if not isinstance(raw, dict):
        raise SweepConfigError("top level must be a mapping")
    experiments_raw = raw.pop("experiments", None)
    if not isinstance(experiments_raw, dict) or not experiments_raw:
        raise SweepConfigError("experiments must be a non-empty mapping name -> settings")
    global_raw = {k: v for k, v in raw.items() if k in GLOBAL_KEYS}
    defaults = _parse_matrix("top level", {k: v for k, v in raw.items()
                                           if k not in GLOBAL_KEYS})
    for key, (var, kind, _) in GLOBAL_KEYS.items():
        if env is not None and var in env:
            global_raw[key] = lsc._env_value(kind, env[var])
    cfg: dict[str, object] = {key: default for key, (_, _, default) in GLOBAL_KEYS.items()}
    for key, value in global_raw.items():
        cfg[key] = lsc._parse_value(key, GLOBAL_KEYS[key][1], value)

    experiments: dict[str, dict[str, object]] = {}
    for name, body in experiments_raw.items():
        if not isinstance(name, str) or not lsc.NAME.match(name):
            raise SweepConfigError(f"invalid experiment name {name!r} (letters, digits, . _ -)")
        if not isinstance(body, dict) or "type" not in body:
            raise SweepConfigError(f"experiment {name}: needs a type ({', '.join(TYPES)})")
        body = dict(body)
        etype = body.pop("type")
        if etype not in TYPES:
            raise SweepConfigError(f"experiment {name}: type {etype!r} not in {list(TYPES)}")
        exp = {key: default for key, (_, default) in MATRIX_KEYS.items()}
        exp.update(defaults)
        exp.update(_parse_matrix(f"experiment {name}", body))
        exp.update({k: cfg[k] for k in GLOBAL_KEYS})
        exp["name"] = name
        exp["type"] = etype
        if exp["plots"] is None:
            exp["plots"] = list(TYPES[etype])
        if exp["overlaps"] is None:
            exp["overlaps"] = [cfg["overlap"]]
        validate(exp)
        experiments[name] = exp
    if only:
        unknown = [n for n in only if n not in experiments]
        if unknown:
            raise SweepConfigError(f"unknown experiment(s) {unknown} (have {list(experiments)})")
        experiments = {n: e for n, e in experiments.items() if n in only}
    cfg["experiments"] = experiments
    return cfg


def validate(exp: dict[str, object]) -> None:
    """Cross-key checks of one experiment; raises ``SweepConfigError``."""
    name = exp["name"]
    bad = [f for f in exp["plots"] if f not in TYPES[exp["type"]]]
    if bad:
        raise SweepConfigError(f"experiment {name}: plots {bad} do not belong to type "
                               f"{exp['type']} ({TYPES[exp['type']]})")
    for key in ("gpus_pencil", "gpus_brick"):
        if any(g < 2 for g in exp[key]):
            raise SweepConfigError(f"experiment {name}: {key} entries must be >= 2 (a 1-rank "
                                   "run has no decomposition)")
    if exp["hp_halo"] == ["on"] and exp["overlaps"] == ["none"]:
        raise SweepConfigError(f"experiment {name}: hp_halo on needs an overlap other than none "
                               "(no halo stream to prioritise otherwise)")
    if "direct" in exp["halo_exchange"] and not (
            set(exp["grids"]) & set(lsc.DIRECT_GRIDS)
            and set(exp["decompositions"]) & set(lsc.DIRECT_DECOMPOSITIONS)):
        raise SweepConfigError(f"experiment {name}: halo_exchange direct only applies to grids "
                               f"{list(lsc.DIRECT_GRIDS)} with decompositions "
                               f"{list(lsc.DIRECT_DECOMPOSITIONS)}")
    if not runs(exp):
        raise SweepConfigError(f"experiment {name}: the grids x gpus matrix is empty")


def runs(exp: dict[str, object]) -> list[tuple[str, str, str, str, int]]:
    """``(grid, decomposition, hp_halo, overlap, gpus)`` of every run, sizes aside.

    - ``dense`` / ``disg`` run in one process on every ``gpus`` entry (z split across the
      GPUs); decomposition is ``slab``.
    - ``dense_distributed`` / ``disg_distributed`` run one MPI rank per GPU, each
      decomposition on its own GPU counts (``gpus_of``), skipping G=1;
      ``halo_exchange: [.., direct]`` adds ``pencil-direct`` / ``brick-direct`` for
      ``dense_distributed`` (``decomposition_variants``).
    - every run is done once per ``overlaps`` entry, except hp-halo on with overlap ``none``.
    """
    out = []
    for grid in exp["grids"]:
        distributed = grid in DISTRIBUTED_GRIDS
        decomps = decomposition_variants(exp, grid) if distributed else ["slab"]
        for decomp in decomps:
            gpus = gpus_of(exp, decomp) if distributed else exp["gpus"]
            for hp in exp["hp_halo"]:
                for ov in exp["overlaps"]:
                    if hp == "on" and ov == "none":
                        continue
                    for g in gpus:
                        if distributed and g == 1:
                            continue
                        out.append((grid, decomp, hp, ov, g))
    return out


def label(grid: str, decomposition: str, hp_halo: str, ov_tag: str = "") -> str:
    """Report label of one run, the ``<label>`` in ``cg-<label>-n<N>-g<G>.json``:
    ``<grid>`` for single-process grids (``-hpon`` with hp-halo on), else
    ``<grid>-<decomposition>-hp<off|on>``; ``-ov<mode>`` for a run whose overlap is not the
    global one (``ov_tag``, see ``overlap_tag``)."""
    ov = f"-ov{ov_tag}" if ov_tag else ""
    if grid in DISTRIBUTED_GRIDS:
        return f"{grid}-{decomposition}-hp{hp_halo}{ov}"
    return grid + ("-hpon" if hp_halo == "on" else "") + ov


def all_runs(cfg: dict[str, object]) -> list[tuple[str, ...]]:
    """Every run of every selected experiment, once: ``(grid, decomposition, hp_halo,
    overlap, gpus, n, label)``, in experiment order."""
    seen: dict[tuple[str, ...], None] = {}
    for exp in cfg["experiments"].values():
        for grid, decomp, hp, ov, g in runs(exp):
            for n in exp["sizes"]:
                run_label = label(grid, decomp, hp, overlap_tag(exp, ov))
                seen.setdefault((grid, decomp, hp, ov, str(g), str(n), run_label), None)
    return list(seen)


def experiment_run_count(exp: dict[str, object]) -> int:
    return len(runs(exp)) * len(exp["sizes"])


def read_sweep_config(path: Path, only: list[str] | None = None) -> dict[str, object]:
    """Load a sweep YAML file; the path must be an existing ``.yml`` / ``.yaml`` file."""
    path = Path(path).resolve()
    if path.suffix not in (".yml", ".yaml") or not path.is_file():
        raise SweepConfigError(f"not a .yml/.yaml file: {path}")
    return load_sweep_config(path.read_text(), only=only)


def main() -> int:
    try:
        cfg = load_sweep_config(os.environ.get("CG_SWEEP_CONFIG", ""), dict(os.environ),
                                split_names(os.environ.get("CG_SWEEP_ONLY")))
    except SweepConfigError as e:
        sys.stderr.write(f"error: config: {e}\n")
        return 1
    for key, (var, kind, _) in GLOBAL_KEYS.items():
        if var in os.environ:
            sys.stderr.write(f"==> {var} taken from the environment\n")
        print(f"{var}={lsc._shell_value(kind, cfg[key])}")
    print("EXPERIMENTS=" + " ".join(f"{e['name']}:{e['type']}:{experiment_run_count(e)}"
                                    for e in cfg["experiments"].values()))
    print("RUNS=" + " ".join(":".join(r) for r in all_runs(cfg)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
