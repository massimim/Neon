#!/usr/bin/env python3
"""Plot CG strong scaling as MDOF/s per GPU.

Reads the ``carbon-cg-report-v1`` reports of a ``cg-sweep-run.sh`` sweep
(``<data_dir>/json/cg-<label>-n<N>-g<G>.json``) and, for every experiment, writes the
figures of its ``plots`` key into ``<data_dir>/plots/<experiment>/``:

  vs_gpus         cg-mdofs-per-gpu-vs-gpus-n<N>.png  MDOF/s per GPU vs GPU count, one panel
                                                     per grid x decomposition, one curve per
                                                     hp-halo x overlap; the dashed ideal
                                                     line is the 1-GPU reference of size N
                                                     (cg-<reference_label>-n<N>-g1.json)
  direct_speedup  cg-direct-speedup.png              MDOF/s(direct) / MDOF/s(staged) vs N, one
                                                     panel per decomposition with both, one
                                                     curve per GPU count x hp-halo x overlap

MDOF/s is ``results.mdofs``: N^3 cells x timed iterations / timed seconds / 1e6, one
iteration being the matvec, both dot products and the vector updates. Panel layout,
axes and styles are the LBM plotter's (``plot_scaling.py``).

Example::

    python3 benchmarks/cg/scripts/plot_cg_scaling.py \\
        --config benchmarks/cg/scripts/cg-sweep.yml
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "lbm_ldc" / "scripts" / "plotting"))
import cg_sweep_config as sc  # noqa: E402
import lbm_reports as lr  # noqa: E402
import plot_scaling as ps  # noqa: E402

SCHEMA = "carbon-cg-report-v1"
FIGURES = ("vs_gpus", "direct_speedup")
MDOFS_PER_GPU_LABEL = "MDOF/s per GPU (10\u2076 cell updates / s / GPU)"
HP_COLORS = {"off": "#1f77b4", "on": "#d62728"}
OVERLAP_COLOR = "#2ca02c"

Panel = tuple[str, str]  # (grid, decomposition)


@dataclass(frozen=True)
class Report:
    path: Path
    gpus: int
    raw: dict

    @property
    def mdofs(self) -> float:
        return float(self.raw["results"]["mdofs"])

    @property
    def mdofs_per_gpu(self) -> float:
        return self.mdofs / self.gpus

    @property
    def process_grid(self) -> str | None:
        dims = self.raw.get("config", {}).get("process_grid")
        return "x".join(str(v) for v in dims) if dims else None


def load(json_dir: Path, label: str, n: int, g: int) -> Report | None:
    """The report of one run, or None when it is missing or not a v1 CG report."""
    path = json_dir / f"{sc.SERIES}-{label}-n{n}-g{g}.json"
    if not path.is_file():
        return None
    with path.open() as f:
        raw = json.load(f)
    if raw.get("schema") != SCHEMA:
        sys.stderr.write(f"warning: skipping {path.name}: bad schema\n")
        return None
    return Report(path, g, raw)


class Series:
    """Every report of one experiment, indexed by (label, n, g), plus the 1-GPU references.
    Has the methods ``plot_scaling.gpu_axis`` uses."""

    def __init__(self, json_dir: Path, cfg: dict) -> None:
        self.cfg = cfg
        self.overlap = cfg["overlap"]
        self.cells = sc.runs(cfg)
        self.ref_name = cfg["reference_label"]
        self.reports: dict[tuple[str, int, int], Report] = {}
        self.refs: dict[int, Report] = {}
        for n in cfg["sizes"]:
            ref = load(json_dir, self.ref_name, n, 1)
            if ref is not None:
                self.refs[n] = ref
            for grid, decomp, hp, ov, g in self.cells:
                label = sc.label(grid, decomp, hp, sc.overlap_tag(cfg, ov))
                rep = load(json_dir, label, n, g)
                if rep is not None:
                    self.reports[(label, n, g)] = rep

    def panels(self) -> list[Panel]:
        return list(dict.fromkeys((grid, decomp) for grid, decomp, *_ in self.cells))

    def curves(self, panel: Panel) -> list[tuple[str, str]]:
        """``(hp_halo, overlap)`` of every curve of a panel."""
        return list(dict.fromkeys((hp, ov) for grid, decomp, hp, ov, _ in self.cells
                                  if (grid, decomp) == panel))

    def gpus(self, panel: Panel) -> list[int]:
        return sorted({g for grid, decomp, *_, g in self.cells if (grid, decomp) == panel})

    def max_gpus(self) -> int:
        return max(g for *_, g in self.cells)

    def get(self, panel: Panel, hp: str, ov: str, n: int, g: int) -> Report | None:
        return self.reports.get((sc.label(*panel, hp, sc.overlap_tag(self.cfg, ov)), n, g))

    def process_grid(self, panel: Panel, g: int) -> str | None:
        for hp, ov in self.curves(panel):
            for n in self.cfg["sizes"]:
                rep = self.get(panel, hp, ov, n, g)
                if rep is not None:
                    return rep.process_grid
        return None

    def describe(self) -> str:
        rep = next(iter(self.reports.values()))
        cfg = rep.raw["config"]
        gpus = rep.raw["system"].get("gpus", [])
        overlap = "per curve" if len(self.cfg["overlaps"]) > 1 else cfg["overlap"]
        text = (f"CG Poisson fp64, {cfg['warmup']} warmup + {cfg['max_iter'] - cfg['warmup']} "
                f"timed iterations, overlap={overlap}")
        return text + (f", {gpus[0]['name']}" if gpus else "")


def curve_label(hp: str, ov_tag: str) -> str:
    return f"hp-halo {hp}" + (f", overlap {ov_tag}" if ov_tag else "")


def curve_kwargs(hp: str, ov_tag: str) -> dict:
    marker, ls = ps.curve_style(hp, ov_tag)
    color = OVERLAP_COLOR if ov_tag else HP_COLORS[hp]
    if ov_tag and hp == "on":
        color = "#9467bd"
    return dict(marker=marker, ls=ls, lw=1.8, ms=5, color=color, label=curve_label(hp, ov_tag))


def subplots(n_panels: int, ylabel: str):
    fig, axes = ps.subplots(n_panels)
    for ax in axes:
        if ax.get_ylabel():
            ax.set_ylabel(ylabel)
    return fig, axes


def plot_vs_gpus(s: Series, n: int, output: Path, show: bool) -> bool:
    ref = s.refs.get(n)
    if ref is None:
        sys.stderr.write(f"warning: {s.cfg['name']}: no 1-GPU reference ({s.ref_name}) for "
                         f"N={n}, skipping\n")
        return False
    ref_v = ref.mdofs
    panels = s.panels()
    fig, axes = subplots(len(panels), MDOFS_PER_GPU_LABEL)
    g_max = s.max_gpus()
    has_runs = False
    for ax, panel in zip(axes, panels):
        ax.plot([1, g_max], [ref_v, ref_v], "--", color="#888888", lw=1.0,
                label=f"Ideal (1 GPU: {ref_v:.0f})")
        gpus = s.gpus(panel)
        if 1 not in gpus:
            ax.plot([1], [ref_v], "*", color="black", ms=11, label="1-GPU reference")
        for hp, ov in s.curves(panel):
            pts = {g: rep.mdofs_per_gpu for g in gpus
                   if (rep := s.get(panel, hp, ov, n, g)) is not None}
            if pts:
                has_runs = True
                xs = sorted(pts)
                ax.plot(xs, [pts[x] for x in xs], **curve_kwargs(hp, sc.overlap_tag(s.cfg, ov)))
        ax.set_title(ps.panel_title(panel))
        ax.set_ylim(bottom=0.0, top=max(ax.get_ylim()[1], ref_v * 1.15))
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend(loc="lower left", fontsize=8)
        ps.gpu_axis(ax, s, panel, sorted({1, *gpus}))
    if not has_runs:
        import matplotlib.pyplot as plt

        plt.close(fig)
        return False
    fig.suptitle(f"Carbon CG strong scaling at N={n}: MDOF/s per GPU\n"
                 f"{s.describe()}; ideal: 1-GPU {s.ref_name}", fontsize=10, y=1.02)
    fig.tight_layout()
    lr.save(fig, output, show)
    return True


def plot_direct_speedup(s: Series, output: Path, show: bool) -> bool:
    import matplotlib.pyplot as plt

    pairs = [(panel, (panel[0], panel[1] + sc.DIRECT_SUFFIX)) for panel in s.panels()
             if (panel[0], panel[1] + sc.DIRECT_SUFFIX) in s.panels()]
    if not pairs:
        return False
    sizes = s.cfg["sizes"]
    cmap = plt.get_cmap("viridis")
    fig, axes = subplots(len(pairs), "MDOF/s(direct halo) / MDOF/s(staged halo)")
    has_runs = False
    for ax, (staged, direct) in zip(axes, pairs):
        gpus = sorted(set(s.gpus(staged)) & set(s.gpus(direct)))
        for i, g in enumerate(gpus):
            for hp, ov in s.curves(direct):
                xs, ys = [], []
                for n in sizes:
                    a, b = s.get(direct, hp, ov, n, g), s.get(staged, hp, ov, n, g)
                    if a is not None and b is not None:
                        xs.append(n)
                        ys.append(a.mdofs / b.mdofs)
                if not xs:
                    continue
                has_runs = True
                tag = sc.overlap_tag(s.cfg, ov)
                marker, ls = ps.curve_style(hp, tag)
                pg = s.process_grid(direct, g)
                ax.plot(xs, ys, marker=marker, ls=ls, lw=1.5, ms=5,
                        color=cmap(0.85 * i / max(1, len(gpus) - 1)),
                        label=f"{g} GPUs" + (f" ({pg})" if pg else "") + ", "
                        + curve_label(hp, tag))
        ax.axhline(1.0, color="gray", ls="--", lw=0.8)
        ax.set_title(f"{ps.panel_title(staged)}: direct / staged")
        ps.size_axis(ax, sizes)
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend(loc="best", fontsize=7)
    if not has_runs:
        plt.close(fig)
        return False
    fig.suptitle(f"Carbon CG direct vs staged halo exchange\n{s.describe()}", fontsize=10,
                 y=1.02)
    fig.tight_layout()
    lr.save(fig, output, show)
    return True


def main() -> int:
    p = argparse.ArgumentParser(description="Plot CG strong scaling as MDOF/s per GPU.")
    lr.add_common_args(p)
    args = p.parse_args()
    try:
        cfg = sc.read_sweep_config(args.config, only=sc.split_names(args.only))
        base = lr.repo_dir(args.data_dir or lr.REPO_ROOT / cfg["data_dir"], must_exist=True)
        json_dir = lr.repo_dir(base / "json", must_exist=True)
    except (sc.SweepConfigError, ValueError) as e:
        sys.stderr.write(f"error: {e}\n")
        return 1
    exps = [e for e in cfg["experiments"].values() if set(e["plots"]) & set(FIGURES)]
    if not exps:
        sys.stderr.write(f"note: no experiment of {args.config} plots {', '.join(FIGURES)}\n")
        return 0
    wrote = False
    for exp in exps:
        out_dir = base / "plots" / exp["name"]
        s = Series(json_dir, exp)
        if not s.reports:
            sys.stderr.write(f"warning: {exp['name']}: no reports in {json_dir}\n")
            continue
        if "vs_gpus" in exp["plots"]:
            for n in exp["sizes"]:
                wrote |= plot_vs_gpus(s, n, out_dir / f"cg-mdofs-per-gpu-vs-gpus-n{n}.png",
                                      args.show)
        if "direct_speedup" in exp["plots"]:
            wrote |= plot_direct_speedup(s, out_dir / "cg-direct-speedup.png", args.show)
    return 0 if wrote else 1


if __name__ == "__main__":
    sys.exit(main())
