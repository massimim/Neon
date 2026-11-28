#!/usr/bin/env python3
"""Plot LBM strong scaling as MLUPS per GPU.

One panel per grid (``dense``, ``disg``) or distributed grid x decomposition
(``dense_distributed`` slab | pencil | brick, ...), one curve per transport x hp-halo x
overlap (an overlap other than the global one is drawn dotted with square markers).
The y axis is always ``results.mlups / G``, so perfect strong scaling is a horizontal
line: the dashed "ideal" line is the 1-GPU reference run of the same size N.

``--config`` is a sweep YAML of ``lbm-sweep-run.sh``. For every ``strong_scaling`` and
``domain_size`` experiment and every configs x precisions series
``<series> = <lattice>-<collision>-<precision>`` it writes the figures of the
experiment's ``plots`` key into ``<data_dir>/plots/<experiment>/``:

  strong_scaling
    vs_gpus     <series>-mlups-per-gpu-vs-gpus-n<N>.png  MLUPS/GPU vs GPU count, per size N
    hp_speedup  <series>-hp-speedup-<t>.png              MLUPS(hp-halo on) / MLUPS(hp-halo
                                                         off) vs GPU count, one curve per
                                                         N (when hp_halo has both)
    grids       <series>-grids-n<N>.png                  MLUPS/GPU vs GPU count, one curve
                                                         per grid: left the same transport
                                                         (NCCL, slab, hp-halo off), right
                                                         each grid's fastest run per GPU
                                                         count, tagged with its
                                                         configuration (two or more grids)
    direct_speedup <series>-direct-speedup.png           MLUPS(direct) / MLUPS(staged) vs N,
                                                         one panel per decomposition with
                                                         both, one curve per GPU count x
                                                         hp-halo x overlap
  domain_size
    vs_n        <series>-mlups-per-gpu-vs-n-g<G>.png     MLUPS/GPU vs N at the largest GPU
                                                         count G, against the per-N
                                                         reference

The reference of size N is ``<series>-<reference_label>-n<N>-g1.json`` in the same
``json/`` folder (default ``dense-intra``). Distributed panels label each GPU count with
the process grid ProcessGrid chose (x*y*z).

Example::

    python3 benchmarks/lbm_ldc/scripts/plotting/plot_scaling.py \\
        --config benchmarks/lbm_ldc/scripts/lbm-sweep.yml --only grids
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import lbm_reports as lr  # noqa: E402
import lbm_sweep_config as sc  # noqa: E402

Panel = tuple[str, str]  # (grid, decomposition)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot LBM strong scaling as MLUPS per GPU.")
    lr.add_common_args(p)
    return p.parse_args()


class Series:
    """Every report of one series, indexed by (label, n, g), plus its 1-GPU references."""

    def __init__(self, name: str, report_dir: Path, cfg: dict,
                 reference: tuple[Path, str, str]) -> None:
        self.name = name
        self.cfg = cfg
        self.overlap = cfg["overlap"]
        self.cells = sc.runs(cfg)
        ref_dir, ref_series, ref_label = reference
        self.ref_name = ref_label
        self.reports: dict[tuple[str, int, int], lr.Report] = {}
        self.refs: dict[int, lr.Report] = {}
        for n in cfg["sizes"]:
            ref = lr.load(ref_dir, ref_series, ref_label, n, 1)
            if ref is not None:
                self.refs[n] = ref
            for grid, decomp, t, hp, ov, g in self.cells:
                label = sc.label(grid, decomp, t, hp, sc.overlap_tag(cfg, ov))
                rep = lr.load(report_dir, name, label, n, g)
                if rep is not None:
                    self.reports[(label, n, g)] = rep

    def panels(self) -> list[Panel]:
        return list(dict.fromkeys((grid, decomp) for grid, decomp, *_ in self.cells))

    def curves(self, panel: Panel) -> list[tuple[str, str, str]]:
        """``(transport, hp_halo, overlap)`` of every curve of a panel."""
        return list(dict.fromkeys((t, hp, ov) for grid, decomp, t, hp, ov, _ in self.cells
                                  if (grid, decomp) == panel))

    def gpus(self, panel: Panel) -> list[int]:
        return sorted({g for grid, decomp, *_, g in self.cells if (grid, decomp) == panel})

    def max_gpus(self) -> int:
        return max(g for *_, g in self.cells)

    def get(self, panel: Panel, t: str, hp: str, n: int, g: int,
            ov: str | None = None) -> lr.Report | None:
        """The report of one cell; ``ov`` defaults to the global overlap."""
        tag = sc.overlap_tag(self.cfg, self.overlap if ov is None else ov)
        return self.reports.get((sc.label(*panel, t, hp, tag), n, g))

    def ref_mlups(self, n: int) -> float | None:
        ref = self.refs.get(n)
        return None if ref is None else ref.mlups

    def process_grid(self, panel: Panel, g: int) -> str | None:
        labels = {sc.label(*panel, t, hp, sc.overlap_tag(self.cfg, ov))
                  for t, hp, ov in self.curves(panel)}
        for (label, _, gg), rep in self.reports.items():
            if gg == g and label in labels:
                return rep.process_grid
        return None

    def describe(self) -> str:
        overlap = "per curve" if len(self.cfg["overlaps"]) > 1 else None
        return lr.describe(next(iter(self.reports.values())), overlap)


def panel_title(panel: Panel) -> str:
    grid, decomp = panel
    if grid in sc.DISTRIBUTED_GRIDS:
        return f"{lr.GRID_TITLES[grid]}, {lr.DECOMP_TITLES[decomp]}"
    return lr.GRID_TITLES[grid]


def curve_label(t: str, hp: str, ov_tag: str = "") -> str:
    return (lr.TRANSPORT_NAMES[t] + (" + hp-halo" if hp == "on" else "")
            + (f", overlap {ov_tag}" if ov_tag else ""))


def subplots(n_panels: int):
    import matplotlib.pyplot as plt

    ncols = n_panels if n_panels <= 3 else (2 if n_panels == 4 else 3)
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.3 * nrows), sharey=True,
                             squeeze=False)
    flat = list(axes.flat)
    for ax in flat[n_panels:]:
        ax.set_visible(False)
    for row in axes:
        row[0].set_ylabel(lr.MLUPS_PER_GPU_LABEL)
    return fig, flat[:n_panels]


def curve_style(hp: str, ov_tag: str) -> tuple[str, str]:
    """``(marker, linestyle)``: by hp-halo, dotted squares for a non-default overlap."""
    return lr.OVERLAP_STYLE if ov_tag else lr.HP_STYLES[hp]


def plot_curve(ax, pts: dict[int, float], t: str, hp: str, ov_tag: str = "") -> None:
    xs = sorted(pts)
    marker, ls = curve_style(hp, ov_tag)
    ax.plot(xs, [pts[x] for x in xs], marker=marker, ls=ls, lw=1.8, ms=5,
            color=lr.TRANSPORT_COLORS[t], label=curve_label(t, hp, ov_tag))


def gpu_axis(ax, s: Series, panel: Panel, gpus: list[int]) -> None:
    ax.set_xticks(gpus)
    if panel[0] in sc.DISTRIBUTED_GRIDS:
        labels = []
        for g in gpus:
            pg = "ref" if g == 1 else s.process_grid(panel, g)
            labels.append(f"{g}\n{pg}" if pg else str(g))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlabel("Number of GPUs (process grid x*y*z)")
    else:
        ax.set_xlabel("Number of GPUs")
    ax.set_xlim(0.5, s.max_gpus() + 0.5)


def size_axis(ax, sizes: list[int]) -> None:
    many = len(sizes) > 7
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{n}\u00b3" for n in sizes], rotation=45 if many else 0,
                       ha="right" if many else "center", fontsize=8 if many else 10)
    ax.set_xlabel("Domain size")


def plot_vs_gpus(s: Series, n: int, output: Path, show: bool) -> bool:
    ref = s.ref_mlups(n)
    if ref is None:
        sys.stderr.write(f"warning: {s.name}: no 1-GPU reference ({s.ref_name}) for N={n}, "
                         "skipping\n")
        return False
    panels = s.panels()
    fig, axes = subplots(len(panels))
    g_max = s.max_gpus()
    for ax, panel in zip(axes, panels):
        ax.plot([1, g_max], [ref, ref], "--", color="#888888", lw=1.0,
                label=f"Ideal (1 GPU: {ref:.0f})")
        gpus = s.gpus(panel)
        if 1 not in gpus:
            ax.plot([1], [ref], "*", color="black", ms=11, label="1-GPU reference")
        for t, hp, ov in s.curves(panel):
            pts = {g: rep.mlups_per_gpu for g in gpus
                   if (rep := s.get(panel, t, hp, n, g, ov)) is not None}
            if pts:
                plot_curve(ax, pts, t, hp, sc.overlap_tag(s.cfg, ov))
        ax.set_title(panel_title(panel))
        ax.set_ylim(bottom=0.0, top=max(ax.get_ylim()[1], ref * 1.15))
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend(loc="lower left", fontsize=8)
        gpu_axis(ax, s, panel, sorted({1, *gpus}))
    fig.suptitle(f"Carbon LBM strong scaling at N={n}: MLUPS per GPU\n"
                 f"{s.describe()}; ideal: 1-GPU {s.ref_name}", fontsize=10, y=1.02)
    fig.tight_layout()
    lr.save(fig, output, show)
    return True


def plot_vs_n(s: Series, g: int, output: Path, show: bool) -> bool:
    sizes = [n for n in s.cfg["sizes"] if s.ref_mlups(n) is not None]
    if not sizes:
        sys.stderr.write(f"warning: {s.name}: no 1-GPU reference ({s.ref_name}) reports, "
                         "skipping\n")
        return False
    panels = s.panels()
    fig, axes = subplots(len(panels))
    for ax, panel in zip(axes, panels):
        ax.plot(sizes, [s.ref_mlups(n) for n in sizes], "--", color="#888888", lw=1.2,
                marker=".", label="Ideal (1-GPU reference)")
        has_runs = False
        for t, hp, ov in s.curves(panel):
            pts = {n: rep.mlups_per_gpu for n in sizes
                   if (rep := s.get(panel, t, hp, n, g, ov)) is not None}
            if pts:
                has_runs = True
                plot_curve(ax, pts, t, hp, sc.overlap_tag(s.cfg, ov))
        pg = s.process_grid(panel, g) if panel[0] in sc.DISTRIBUTED_GRIDS else None
        ax.set_title(panel_title(panel) + (f", {pg}" if pg else ""))
        size_axis(ax, sizes)
        ax.set_ylim(bottom=0.0)
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend(loc="best", fontsize=8)
        if not has_runs:
            ax.text(0.5, 0.5, f"no runs at {g} GPUs", ha="center", transform=ax.transAxes)
    fig.suptitle(f"Carbon LBM MLUPS per GPU at {g} GPUs vs domain size\n"
                 f"{s.describe()}; ideal: 1-GPU {s.ref_name}", fontsize=10, y=1.02)
    fig.tight_layout()
    lr.save(fig, output, show)
    return True


def plot_hp_speedup(s: Series, out_dir: Path, show: bool) -> bool:
    import matplotlib.pyplot as plt

    if set(s.cfg["hp_halo"]) != {"off", "on"}:
        return False
    cmap = plt.get_cmap("viridis")
    sizes = s.cfg["sizes"]
    panels = s.panels()
    wrote = False
    for t in s.cfg["transports"]:
        tp = [p for p in panels if (t, "on", s.overlap) in s.curves(p)]
        if not tp:
            continue
        fig, axes = subplots(len(tp))
        for ax, panel in zip(axes, tp):
            for i, n in enumerate(sizes):
                xs, ys = [], []
                for g in s.gpus(panel):
                    on, off = s.get(panel, t, "on", n, g), s.get(panel, t, "off", n, g)
                    if on is not None and off is not None:
                        xs.append(g)
                        ys.append(on.mlups / off.mlups)
                if xs:
                    ax.plot(xs, ys, "D-", lw=1.5, ms=5, label=f"N={n}",
                            color=cmap(0.85 * i / max(1, len(sizes) - 1)))
            ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.7)
            ax.set_title(panel_title(panel))
            ax.grid(True, ls=":", alpha=0.5)
            ax.legend(loc="upper left", fontsize=8)
            gpu_axis(ax, s, panel, s.gpus(panel))
        for ax in axes[::min(3, len(tp))]:
            ax.set_ylabel("MLUPS(hp-halo on) / MLUPS(hp-halo off)")
        fig.suptitle(f"High-priority halo stream speedup, {lr.TRANSPORT_NAMES[t]}\n"
                     f"{s.describe()}", fontsize=10, y=1.02)
        fig.tight_layout()
        lr.save(fig, out_dir / f"{s.name}-hp-speedup-{t}.png", show)
        wrote = True
    return wrote


TRANSPORT_TAGS = {"intra": "P", "mpi": "N", "mpip2p": "M"}


def best_tag(grid: str, decomp: str, t: str, hp: str, ov_tag: str = "") -> str:
    tag = TRANSPORT_TAGS[t]
    if grid in sc.DISTRIBUTED_GRIDS:
        base, mode = sc.split_decomposition(decomp)
        tag += base[0] + ("d" if mode == "direct" else "")
    return tag + ("+hp" if hp == "on" else "") + (f"/{ov_tag}" if ov_tag else "")


def plot_grids(s: Series, n: int, output: Path, show: bool) -> bool:
    import matplotlib.pyplot as plt

    grids = list(dict.fromkeys(grid for grid, *_ in s.cells))
    if len(grids) < 2:
        return False
    transports = s.cfg["transports"]
    t0 = "mpi" if "mpi" in transports else next((t for t in transports if t != "intra"),
                                                 "intra")
    hp0 = "off" if "off" in s.cfg["hp_halo"] else s.cfg["hp_halo"][0]
    ref = s.ref_mlups(n)
    g_max = s.max_gpus()
    fig = plt.figure(figsize=(12, 6.2))
    layout = fig.add_gridspec(2, 2, height_ratios=[4.0, 1.2], hspace=0.35)
    same = fig.add_subplot(layout[0, 0])
    best = fig.add_subplot(layout[0, 1], sharey=same)
    key = fig.add_subplot(layout[1, :])
    key.axis("off")
    has_runs = False
    tags: dict[str, dict[int, str]] = {}
    for grid in grids:
        color, marker = lr.GRID_STYLES[grid]
        style = dict(color=color, marker=marker, lw=1.8, ms=5, label=lr.GRID_TITLES[grid])
        panel = (grid, "slab")
        if (t0, hp0, s.overlap) in s.curves(panel):
            pts = {g: rep.mlups_per_gpu for g in s.gpus(panel)
                   if (rep := s.get(panel, t0, hp0, n, g)) is not None}
            if pts:
                xs = sorted(pts)
                same.plot(xs, [pts[g] for g in xs], **style)
                has_runs = True
        winners: dict[int, tuple[float, str]] = {}
        for gr, decomp, t, hp, ov, g in s.cells:
            if gr != grid:
                continue
            rep = s.get((gr, decomp), t, hp, n, g, ov)
            if rep is not None and rep.mlups_per_gpu > winners.get(g, (0.0, ""))[0]:
                winners[g] = (rep.mlups_per_gpu,
                              best_tag(gr, decomp, t, hp, sc.overlap_tag(s.cfg, ov)))
        if winners:
            xs = sorted(winners)
            best.plot(xs, [winners[g][0] for g in xs], **style)
            tags[grid] = {g: winners[g][1] for g in xs}
            has_runs = True
    if not has_runs:
        plt.close(fig)
        return False
    for ax in (same, best):
        if ref is not None:
            ax.plot([1, g_max], [ref, ref], "--", color="#888888", lw=1.0,
                    label=f"Ideal (1 GPU: {ref:.0f})")
        ax.set_xticks(range(1, g_max + 1))
        ax.set_xlim(0.5, g_max + 0.5)
        ax.set_xlabel("Number of GPUs")
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend(loc="lower left", fontsize=8)
    same.set_ylabel(lr.MLUPS_PER_GPU_LABEL + "\n(axis not from zero)")
    same.set_title(f"Same transport: {lr.TRANSPORT_NAMES[t0]}, slab"
                   + (", hp-halo on" if hp0 == "on" else ""))
    best.set_title("Fastest configuration per GPU count")
    plt.setp(best.get_yticklabels(), visible=False)
    gpus = list(range(1, g_max + 1))
    table = key.table(
        cellText=[[tags[grid].get(g, "") for g in gpus] for grid in tags],
        rowLabels=[lr.GRID_TITLES[grid] for grid in tags],
        rowColours=[lr.GRID_STYLES[grid][0] + "40" for grid in tags],
        colLabels=[f"{g} GPU" + ("s" if g > 1 else "") for g in gpus],
        cellLoc="center", bbox=[0.22, 0.12, 0.78, 0.88])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    key.text(0.0, 1.0, "Fastest configuration", fontsize=9, weight="bold", va="bottom",
             transform=key.transAxes)
    key.text(0.0, 0.0, "P = Peeraccess, N = NCCL, M = MPI (P2P); s / p / b = slab / pencil / "
             "brick (distributed grids), pd / bd = pencil / brick with direct halo; +hp = hp-halo on; "
             "/mode = overlap mode when not the sweep's",
             fontsize=8, color="#555555",
             transform=key.transAxes)
    fig.suptitle(f"Carbon LBM grids at N={n}: MLUPS per GPU\n{s.describe()}"
                 + (f"; ideal: 1-GPU {s.ref_name}" if ref is not None else ""),
                 fontsize=10)
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
    fig, axes = subplots(len(pairs))
    has_runs = False
    for ax, (staged, direct) in zip(axes, pairs):
        gpus = sorted(set(s.gpus(staged)) & set(s.gpus(direct)))
        for i, g in enumerate(gpus):
            for t, hp, ov in s.curves(direct):
                xs, ys = [], []
                for n in sizes:
                    a, b = s.get(direct, t, hp, n, g, ov), s.get(staged, t, hp, n, g, ov)
                    if a is not None and b is not None:
                        xs.append(n)
                        ys.append(a.mlups / b.mlups)
                if not xs:
                    continue
                has_runs = True
                tag = sc.overlap_tag(s.cfg, ov)
                marker, ls = curve_style(hp, tag)
                pg = s.process_grid(direct, g)
                ax.plot(xs, ys, marker=marker, ls=ls, lw=1.5, ms=5,
                        color=cmap(0.85 * i / max(1, len(gpus) - 1)),
                        label=f"{g} GPUs" + (f" ({pg})" if pg else "") + ", "
                        + curve_label(t, hp, tag))
        ax.axhline(1.0, color="gray", ls="--", lw=0.8)
        ax.set_title(f"{panel_title(staged)}: direct / staged")
        size_axis(ax, sizes)
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend(loc="best", fontsize=7)
    if not has_runs:
        plt.close(fig)
        return False
    for ax in axes[::min(3, len(pairs))]:
        ax.set_ylabel("MLUPS(direct halo) / MLUPS(staged halo)")
    fig.suptitle(f"Carbon LBM direct vs staged halo exchange\n{s.describe()}", fontsize=10,
                 y=1.02)
    fig.tight_layout()
    lr.save(fig, output, show)
    return True


def main() -> int:
    args = parse_args()
    try:
        exps, json_dir, plots_dir = lr.experiments(
            args, ("vs_gpus", "hp_speedup", "grids", "vs_n", "direct_speedup"))
    except (sc.SweepConfigError, ValueError) as e:
        sys.stderr.write(f"error: {e}\n")
        return 1
    if not exps:
        return 0
    wrote = False
    for exp in exps:
        out_dir = plots_dir / exp["name"]
        plots = exp["plots"]
        for series, *_ in sc.series_prefixes(exp):
            s = Series(series, json_dir, exp, (json_dir, series, exp["reference_label"]))
            if not s.reports:
                sys.stderr.write(f"warning: {exp['name']}: no reports for {series} in "
                                 f"{json_dir}\n")
                continue
            if "vs_gpus" in plots:
                for n in exp["sizes"]:
                    wrote |= plot_vs_gpus(
                        s, n, out_dir / f"{series}-mlups-per-gpu-vs-gpus-n{n}.png", args.show)
            if "vs_n" in plots:
                g = s.max_gpus()
                wrote |= plot_vs_n(s, g, out_dir / f"{series}-mlups-per-gpu-vs-n-g{g}.png",
                                   args.show)
            if "hp_speedup" in plots:
                wrote |= plot_hp_speedup(s, out_dir, args.show)
            if "grids" in plots:
                for n in exp["sizes"]:
                    wrote |= plot_grids(s, n, out_dir / f"{series}-grids-n{n}.png", args.show)
            if "direct_speedup" in plots:
                wrote |= plot_direct_speedup(s, out_dir / f"{series}-direct-speedup.png",
                                             args.show)
    return 0 if wrote else 1


if __name__ == "__main__":
    sys.exit(main())
