#!/usr/bin/env python3
"""Plot the rank x message transfer matrix written by ``lbm_ldc --export-transfers``.

The CSV (``rank,index,seq,kernel,group,op,peer,elements,bytes``) lists every cross-rank halo
message of one skeleton iteration, read from the compiled ``carbon::Skeleton``; the
``# key=value`` lines above its header (grid, decomposition, process grid, N, ...) make
the first line of the title. The plot
has one row per rank and one column per message in issue order; the colour is the
message size and each cell reads ``S>p`` (send to rank p) or ``R<p`` (receive from
rank p) with its size and element count. The right margin gives each rank's message count and bytes sent.
When the CSV names its process grid, a second panel draws the ranks at their process coordinates
(z across, y up, x as depth) with one link per communicating pair, its width the bytes sent per
iteration and its colour the neighbour kind (face, edge, corner). When it also gives ``n``, a third
panel shows the N^3 cavity cut into one subdomain per rank (exploded, same split as
``DenseDistributed``: balanced blocks, the first ``n % parts`` one cell longer).

Example::

    mpiexec -n 8 carbon_benchmark_lbm_ldc --grid disg_distributed --n 512 \\
        --export-transfers transfers.csv --compile-only
    python3 benchmarks/lbm_ldc/scripts/plotting/plot_transfers.py transfers.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import lbm_reports as lr  # noqa: E402
import lbm_sweep_config as sc  # noqa: E402

COLUMNS = ("rank", "index", "seq", "kernel", "group", "op", "peer", "elements", "bytes")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot the rank x message transfer matrix.")
    p.add_argument("csv", type=Path, help="CSV written by --export-transfers")
    p.add_argument("--out", type=Path, default=None,
                   help="output PNG (default: the CSV path with a .png suffix)")
    return p.parse_args()


def checked_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    """Resolve and validate the command-line paths: an existing .csv in, a .png out."""
    src = args.csv.expanduser().resolve()
    if src.suffix.lower() != ".csv" or not src.is_file():
        raise ValueError(f"{args.csv}: not an existing .csv file")
    out = (args.out.expanduser().resolve() if args.out is not None
           else src.with_suffix(".png"))
    if out.suffix.lower() != ".png":
        raise ValueError(f"{out}: output must end in .png")
    if not out.parent.is_dir():
        raise ValueError(f"{out.parent}: output directory does not exist")
    return src, out


def read_messages(src: Path) -> tuple[dict[str, str], list[dict]]:
    """``# key=value`` lines above the header (the run description), then the messages."""
    meta: dict[str, str] = {}
    body: list[str] = []
    with src.open(newline="") as f:
        for line in f:
            if line.startswith("#"):
                key, sep, value = line[1:].strip().partition("=")
                if sep and key.replace("_", "").isalnum():
                    meta[key] = value
            else:
                body.append(line)
    reader = csv.DictReader(body)
    if tuple(reader.fieldnames or ()) != COLUMNS:
        raise ValueError(f"{src}: expected header {','.join(COLUMNS)}")
    rows = []
    for r in reader:
        if r["op"] not in ("send", "recv"):
            raise ValueError(f"{src}: bad op {r['op']!r}")
        rows.append({"rank": int(r["rank"]), "index": int(r["index"]),
                     "seq": int(r["seq"]), "kernel": r["kernel"],
                     "op": r["op"], "peer": int(r["peer"]), "elements": int(r["elements"]),
                     "bytes": int(r["bytes"])})
    return meta, rows


def run_title(meta: dict[str, str]) -> str:
    """One line describing the run, e.g. ``DenseDisgDistributed, slab (z) 1x1x8, N = 512``."""
    if "grid" not in meta:
        return ""
    parts = [lr.GRID_TITLES.get(meta["grid"], meta["grid"])]
    if meta["grid"] in sc.DISTRIBUTED_GRIDS and "decomposition" in meta:
        decomp = lr.DECOMP_TITLES.get(meta["decomposition"], meta["decomposition"])
        parts.append(f"{decomp} {meta.get('process_grid', '')}".strip())
    if "n" in meta:
        parts.append(f"N = {meta['n']}")
    if "ranks" in meta:
        parts.append(f"{meta['ranks']} ranks")
    model = " ".join(meta[k] for k in ("lattice", "collision", "precision") if k in meta)
    if model:
        parts.append(model)
    return ", ".join(parts)


def megabytes(n: int) -> str:
    return f"{n / 1e6:.2f} MB" if n >= 1_000_000 else f"{n / 1e3:.1f} KB"


def process_dims(meta: dict[str, str], ranks: int) -> tuple[int, int, int] | None:
    """``(x, y, z)`` of the ``process_grid=XxYxZ`` line, or None when absent or not matching."""
    parts = meta.get("process_grid", "").split("x")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        return None
    dims = tuple(int(p) for p in parts)
    return dims if dims[0] * dims[1] * dims[2] == ranks else None


def coords_of(rank: int, dims: tuple[int, int, int]) -> tuple[int, int, int]:
    """Process coordinates of a rank; z varies fastest, as in ``carbon::ProcessGrid``."""
    _, dy, dz = dims
    return rank // (dy * dz), (rank // dz) % dy, rank % dz


LINK_KINDS = {1: ("face", "#1f77b4"), 2: ("edge", "#ff7f0e"), 3: ("corner", "#d62728")}
DEPTH = (0.5, 0.42)  # drawing offset of one step in x


def is_slab(dims: tuple[int, int, int]) -> bool:
    """Only z is split: drawn as a column (rank 0 on top, like the matrix rows)."""
    return dims[0] == 1 and dims[1] == 1


def topology_width(dims: tuple[int, int, int]) -> float:
    """Natural width in inches of the topology panel."""
    if is_slab(dims):
        return 4.0
    return max(5.0, 1.8 * (dims[2] + DEPTH[0] * (dims[0] - 1)) + 1.6)


def plot_topology(ax, rows: list[dict], dims: tuple[int, int, int]) -> None:
    """Ranks at their process coordinates, one link per communicating pair."""
    ranks = dims[0] * dims[1] * dims[2]
    slab = is_slab(dims)

    def position(rank: int) -> tuple[float, float]:
        x, y, z = coords_of(rank, dims)
        if slab:
            return 0.0, -1.1 * z
        return z + DEPTH[0] * x, y + DEPTH[1] * x

    sent: dict[tuple[int, int], int] = {}
    for r in rows:
        if r["op"] == "send" and r["peer"] != r["rank"]:
            pair = (min(r["rank"], r["peer"]), max(r["rank"], r["peer"]))
            sent[pair] = sent.get(pair, 0) + r["bytes"]
    max_bytes = max(sent.values(), default=1)
    kinds_seen: set[int] = set()
    for (a, b), nbytes in sorted(sent.items(), key=lambda kv: kv[1]):
        ca, cb = coords_of(a, dims), coords_of(b, dims)
        steps = [abs(p - q) for p, q in zip(ca, cb)]
        kind = min(3, sum(1 for s in steps if s > 0))
        kinds_seen.add(kind)
        wraps = any(s > 1 for s in steps)
        ax.annotate("", xy=position(b), xytext=position(a),
                    arrowprops=dict(arrowstyle="-", lw=1.0 + 6.0 * nbytes / max_bytes,
                                    color=LINK_KINDS[kind][1], alpha=0.75,
                                    shrinkA=0, shrinkB=0,
                                    connectionstyle="arc3,rad=0.35" if wraps else "arc3"))
        if kind == 1:
            (xa, ya), (xb, yb) = position(a), position(b)
            ax.text((xa + xb) / 2 + (0.25 if slab else 0.0), (ya + yb) / 2, megabytes(nbytes),
                    fontsize=8, ha="left" if slab else "center", va="center", color="#333333",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    for rank in range(ranks):
        px, py = position(rank)
        x, y, z = coords_of(rank, dims)
        sep = "  " if slab else "\n"
        ax.text(px, py, f"r{rank}{sep}({x},{y},{z})", ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.35", fc="#e8eef7", ec="#2f3e5c", lw=1.0),
                zorder=3)
    for kind in sorted(kinds_seen):
        name, color = LINK_KINDS[kind]
        ax.plot([], [], color=color, lw=3, label=f"{name} neighbour")
    if kinds_seen:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=len(kinds_seen),
                  fontsize=8, frameon=False)
    xs = [position(r)[0] for r in range(ranks)]
    ys = [position(r)[1] for r in range(ranks)]
    ax.set_xlim(min(xs) - 0.7, max(xs) + (1.6 if slab else 0.7))
    ax.set_ylim(min(ys) - 0.7, max(ys) + 0.7)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    axes = "z \u2193" if slab else "z \u2192, y \u2191, x = depth"
    ax.set_title(f"Process grid {dims[0]}x{dims[1]}x{dims[2]} (one rank per GPU)\n"
                 f"{axes}; link width = bytes sent per iteration", fontsize=9)


def domain_n(meta: dict[str, str], dims: tuple[int, int, int] | None) -> int | None:
    """Cavity side ``n`` when given and large enough to give every rank a cell on each axis."""
    value = meta.get("n", "")
    if dims is None or not value.isdigit():
        return None
    n = int(value)
    return n if n >= max(dims) else None


def split_extent(n: int, parts: int) -> list[int]:
    """Block lengths along one axis, as ``DenseDistributed`` splits it."""
    base, extra = divmod(n, parts)
    return [base + (1 if i < extra else 0) for i in range(parts)]


def extent_text(lengths: list[int]) -> str:
    return str(lengths[0]) if min(lengths) == max(lengths) else f"{min(lengths)}-{max(lengths)}"


def plot_partition(ax, dims: tuple[int, int, int], n: int) -> None:
    """The N^3 cavity cut into one block per rank, pulled apart along each split axis.

    Oriented like the topology panel: a slab stacks z downwards (rank 0 on top), otherwise
    z runs across, y up and x into the page.
    """
    ranks = dims[0] * dims[1] * dims[2]
    slab = is_slab(dims)
    # Domain axis shown on the plot's (horizontal, depth, vertical) axis.
    order = (0, 1, 2) if slab else (2, 0, 1)
    lengths = [split_extent(n, d) for d in dims]
    gap = 0.12 * n
    # Voxel grid with gap cells between blocks: index 2*i is block i of that axis.
    edges = []
    for a in order:
        e = [0.0]
        for i, length in enumerate(lengths[a]):
            if i:
                e.append(e[-1] + gap)
            e.append(e[-1] + length)
        edges.append(np.array(e))
    shape = tuple(2 * dims[a] - 1 for a in order)
    filled = np.zeros(shape, dtype=bool)
    colors = np.zeros(shape + (4,))
    cmap = plt.get_cmap("tab10" if ranks <= 10 else "tab20")
    ax.computed_zorder = False
    for rank in range(ranks):
        c = coords_of(rank, dims)
        idx = tuple(2 * c[a] for a in order)
        filled[idx] = True
        colors[idx] = (*cmap(rank % cmap.N)[:3], 0.55)
        if ranks <= 27:
            centre = [(edges[p][i] + edges[p][i + 1]) / 2 for p, i in enumerate(idx)]
            ax.text(*centre, f"r{rank}", ha="center", va="center", fontsize=8,
                    fontweight="bold", zorder=10,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    gx, gy, gz = np.meshgrid(*edges, indexing="ij")
    ax.voxels(gx, gy, gz, filled, facecolors=colors, edgecolor="#333333", linewidth=0.4)
    ax.set_box_aspect([e[-1] for e in edges])
    if slab:
        ax.invert_zaxis()
    names = "xyz"
    for set_label, a in zip((ax.set_xlabel, ax.set_ylabel, ax.set_zlabel), order):
        set_label(names[a], labelpad=-12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=22, azim=-58)
    sub = " x ".join(extent_text(lengths[a]) for a in range(3))
    ax.set_title(f"Partitioning of the {n}\u00b3 cavity ({dims[0]}x{dims[1]}x{dims[2]} cut)\n"
                 f"subdomain per rank: {sub} cells (x, y, z)", fontsize=9)


def plot(rows: list[dict], title: str, out: Path,
         dims: tuple[int, int, int] | None = None, n: int | None = None) -> None:
    ranks = max(r["rank"] for r in rows) + 1
    cols = max(r["index"] for r in rows) + 1
    grid = [[math.nan] * cols for _ in range(ranks)]
    for r in rows:
        grid[r["rank"]][r["index"]] = r["bytes"] / 1e6

    width = max(7.0, 0.95 * cols + 4.5)
    height = max(3.0, 0.6 * ranks + 1.8)
    if dims is None:
        fig, ax = plt.subplots(figsize=(width, height))
    else:
        ratios = [width, min(topology_width(dims), max(6.0, width))]
        if n is not None:
            ratios.append(5.0)
        fig = plt.figure(figsize=(sum(ratios), height))
        gs = fig.add_gridspec(1, len(ratios), width_ratios=ratios)
        ax = fig.add_subplot(gs[0, 0])
        plot_topology(fig.add_subplot(gs[0, 1]), rows, dims)
        if n is not None:
            plot_partition(fig.add_subplot(gs[0, 2], projection="3d"), dims, n)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#f2f2f2")
    vmax = max(r["bytes"] for r in rows) / 1e6
    im = ax.imshow(grid, cmap=cmap, aspect="auto", interpolation="nearest", vmin=0.0, vmax=vmax)
    for r in rows:
        mb = r["bytes"] / 1e6
        arrow = f"S>{r['peer']}" if r["op"] == "send" else f"R<{r['peer']}"
        ax.text(r["index"], r["rank"], f"{arrow}\n{megabytes(r['bytes'])}\n{r['elements']:,} el", ha="center",
                va="center", fontsize=7, color="black" if mb > 0.6 * vmax else "white")

    totals = []
    for rank in range(ranks):
        mine = [r for r in rows if r["rank"] == rank]
        sent = sum(r["bytes"] for r in mine if r["op"] == "send")
        totals.append(f"{len(mine)} msgs, {megabytes(sent)} sent")

    ax.set_xticks(range(cols), [f"msg {i}" for i in range(cols)], fontsize=8)
    ax.set_yticks(range(ranks), [f"rank {i}" for i in range(ranks)])
    right = ax.secondary_yaxis("right")
    right.set_yticks(range(ranks), totals, fontsize=8)
    ax.set_xlabel("message (issue order)")
    if dims is None:
        ax.set_title(title, fontsize=10)
    else:
        fig.suptitle(title, fontsize=10)
    # pad is a fraction of the axes width; keep ~1.7 in free for the right-hand totals.
    fig.colorbar(im, ax=ax, label="MB", pad=1.7 / (width - 2.8))
    if dims is None:
        fig.tight_layout()
    else:
        # Leave ~0.2 in per title line above the panels for the figure title.
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0 - 0.2 * (title.count("\n") + 1.5) / height))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    try:
        src, out = checked_paths(args)
        meta, rows = read_messages(src)
    except (OSError, ValueError, KeyError) as e:
        sys.stderr.write(f"error: {e}\n")
        return 1
    if not rows:
        print(f"note: {src} has no cross-rank messages, nothing to plot")
        return 0
    kernels = sorted({(r["seq"], r["kernel"]) for r in rows})
    lines = [run_title(meta), "Cross-rank halo messages per skeleton iteration",
             ", ".join(f"#{seq} {name}" for seq, name in kernels)]
    title = "\n".join(line for line in lines if line)
    dims = process_dims(meta, max(r["rank"] for r in rows) + 1)
    plot(rows, title, out, dims, domain_n(meta, dims))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
