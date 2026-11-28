# TODO

## Pencil Boundary single launch (DenseDistributed)

One `dense_pencil_boundary_kernel` launch covers the four Boundary faces of a Pencil span;
`blockIdx.z` selects face and halo layer (`include/carbon/detail/dense_distributed/pencil_boundary.h`).

### Done

- [x] `pencil_boundary.h`: layout, grid, `CARBON_HD` decode; `Span::shell_bands()` shares the band clamps
- [x] `dense_pencil_boundary_kernel` and routing in `launch_dense_grid` (Pencil kind, block `z == 1`)
- [x] `gpu_grid(Boundary)` reports the launched grid for pencil spans
- [x] Host coverage test in `test_dense_distributed_windows.cpp` (1090 pencil spans, 5 block shapes)
- [x] Single-GPU test `test_dense_distributed_pencil_boundary.cu` (PeriodicY, r=1/2, thin z, block overrides)
- [x] 4-rank `test_dense_distributed_dist`: pencil-open / pencil-periodicYZ pass with overlap
- [x] Benchmark at 512^3, pencil 1x2x4, 8 A100, NCCL 2.27.7, `--overlap standard`:
      Boundary GPU time 55.9 -> 29.3 us (LBM), 44.3 -> 22.2 us (CG);
      LBM 57,067 -> 57,560 MLUPS (+0.9 %), CG 2.003 -> 1.986 ms/iter (-0.8 %); results bitwise identical
- [x] CHANGELOG entry

### Open

- [ ] Commit and push on `hp-halo-priority` (changes are uncommitted)
- [ ] `test_dense_distributed_dist` fails on periodic-X cases (`slab-periodicX`, `pencil-periodicX`):
      `pair_count != 0` in the run-based halo check. Pre-existing (baseline `cc70babd` fails identically);
      the test stops at the first failing case, so later cases are skipped in CI
- [x] Outdated comment in `test_dense_distributed_dist.cu` (lines 28-29) about the pencil/brick
      Boundary launch (fixed with the brick change)
- [ ] Optional: same single launch for the SYCL launcher (`sycl/dense_distributed_launch.h`)
- [ ] Optional: same single launch for the reduce stage-1 window loop (`dense_distributed_reduce_launch.h`)
- [x] Brick Boundary: done, see the section below
- [ ] Optional: `--block XxY` flag in lbm_ldc and a sweep of 32x16 / 64x8 / 128x4 / 256x2
      (`DenseDistributed` already accepts `block_override`)
- [ ] `nsys` is not in `carbon-cuda-nccl-pinned:local`; kernel timings were taken with `ncu`

## Brick Boundary single launch (DenseDistributed)

One `dense_brick_boundary_kernel` launch covers the six Boundary faces of a Brick span;
`blockIdx.z` selects face and halo layer, the X faces walk (y, z)
(`include/carbon/detail/dense_distributed/brick_boundary.h`). Separate from the pencil path.

### Done

- [x] `brick_boundary.h`: layout, slices, grid, `CARBON_HD` decode, `uses_brick_boundary_launch`
- [x] `dense_brick_boundary_kernel` and routing in `launch_dense_grid` (Brick kind, block `z == 1`)
- [x] `gpu_grid(Boundary)` reports the launched grid for brick spans
- [x] Host coverage in `test_dense_distributed_windows.cpp` (1,309 brick spans incl. no z / no y halo,
      X faces wider than n.x, translated halo-z-only span; 5 block shapes)
- [x] Single-GPU test `test_dense_distributed_brick_boundary.cu` (1x1x1 PeriodicX / XY / XYZ, r=1/2,
      thin x, thin z, wide X faces, block overrides)
- [x] `CARBON_DIST_CASES=<substring>` filter in `test_dense_distributed_dist`; 4-rank brick cases
      (brickXY / brickXZ, open and periodic) and 8-rank brick 2x2x2 pass with overlap
- [x] LBM and CG brick 8 ranks, N=96 / 101, overlap none / standard: bitwise equal to the baseline
- [x] Benchmark at 512^3, brick 2x2x2, 8 A100, `--overlap standard` (ncu rank 0):
      Boundary 233.8 -> 31.7 us (CG), 368.8 -> 326.9 us (LBM);
      LBM 42,758 -> 43,302 MLUPS (+1.3 %), CG 2.224 -> 2.084 ms/iter (-6.3 %)
- [x] CHANGELOG entry

### Open

- [ ] LBM X faces stay memory-bound: each X-face cell uses 4 B of a 32 B sector per population
      (19 reads + 19 writes). Only a wider X band (Boundary partition change) or a different storage
      layout would help
- [ ] Optional: same single launch for the SYCL launcher and the reduce stage-1 window loop

## HaloDistributed pack/unpack kernel fusion

Same `blockIdx.z` multi-face trick as Pencil Boundary, applied to gather/scatter:
one `PackBatch` launch per axis pass (up to `kMaxPackBatchSteps` steps), so Lower+Upper
(and chunked component lists) share one gather and one scatter. `PackBatch` was later replaced
by the device-memory `PackTable` (see Brick Direct below).

### Done

- [x] `PackBatch` / `make_pack_batch` / `pack_*_batch_thread` in `halo_pack_plan.h`
- [x] `gather_batch_kernel` / `scatter_batch_kernel`; `launch_*_batch`; single-step
      `launch_gather`/`launch_scatter` wrap a 1-step batch (`src/halo_pack.cu`)
- [x] `HaloDistributed::run_pack_steps_` fuses consecutive steps into batch launches
- [x] Host test `tests/test_halo_pack_plan.cpp` (single-step, fused==sequential, chunking, decode)
- [x] ncu on LBM pencil interior rank: gathers/dense=2.00, scatters/dense=2.00
      (was 3+3); Z batch grid_z=20 (= 10+10 comps)

### Measured before implementing (kept for context)

512^3 pencil 1x2x4, `--overlap none`: fusion saves ~2 launches on interior ranks only
(~10 us optimistic, <0.5 % of step). Implemented anyway for simpler launch accounting.
`CARBON_HALO_PACK_Z=0` slowed LBM (57140 vs 58316 MLUPS) — keep Z packing.

- [ ] Optional: cross-pass fusion (Y-scatter with Z-gather / ghost-row forwarding)
      (superseded for Pencil by `HaloExchangeMode::Direct` below)

## Pencil Direct halo exchange (`HaloExchangeMode::Direct`)

Grid-init option: Pencil spans exchange faces + YZ edges with all eight neighbours in one
gather / one NCCL group / one scatter (`build_pencil_direct_pack_plan`).

### Done

- [x] `HaloExchangeMode {Staged, Direct}` on every `DenseDistributed` ctor + `halo_exchange()`
- [x] `PackBatch` switched to a 1D per-step block prefix (mixed Y/Z face shapes, no idle blocks);
      `kMaxPackBatchSteps` 8 -> 16
- [x] `build_pencil_direct_pack_plan`: 8 directions, self wraps, lattice `crosses` filters,
      message key `(direction code, src, 0, dst, 0)`
- [x] `HaloDistributed::build_multi_axis_`: single direct pass; host path keeps staged copies
- [x] Host tests: Direct == staged and == global truth on 10 emulated worlds, D3Q19 lattice
      filters (uni/bidirectional), self-wrap-only world, non-pencil rejection
- [x] GPU: N=96 pencil 1x2x4 LBM/CG, Staged vs Direct, overlap none/standard: bitwise identical;
      slab Direct issues the same messages as Staged (`--export-transfers`)
- [x] ncu: LBM pencil 1 gather + 1 scatter per step (Staged 2 + 2)
- [x] N=256 pencil 1x2x4: LBM +16.7 % (standard) / +14.2 % (none); CG -5.4 % / -5.0 % time
- [x] `--halo-exchange staged|direct` in `lbm_ldc` and `cg_poisson`; CHANGELOG entry

### Open

- [ ] Commit (library + tests + benchmark flags; not the /tmp scripts)
- [ ] Decide whether `Direct` should become the Pencil default
- [ ] Measure at 512^3 and across nodes (edge messages are small: latency-bound off-node)
- [x] Brick Direct (section below)
- [x] GPU case in `test_dense_distributed_dist` (`CARBON_DIST_HALO_EXCHANGE=staged|direct`)
- [ ] SYCL keeps Staged (no packed path)

## Brick Direct halo exchange (`HaloExchangeMode::Direct` on Brick spans)

Brick spans (`halo.x > 0`, packed path) exchange faces, edges and corners with up to 26
neighbours in one gather / one NCCL group / one scatter (`build_brick_direct_pack_plan`).

### Done

- [x] `kBrickDirectDirs`, `direct_neighbour`, `brick_direct_send_box` / `recv_box`,
      `build_brick_direct_pack_plan` (lattice `crosses` filters, self wraps, absent neighbours)
- [x] Thin z-edges packed with y/z swapped (`pack_swap_yz`, `append_oriented_steps`)
- [x] `HaloDistributed` gate: `direct_brick` beside `direct_pencil`
- [x] Host tests: Direct == staged == global truth on x-split worlds, D3Q19/D3Q27 filters,
      thin-edge swap, self-only, `halo.x == 0` reject
- [x] GPU: 8 ranks + 4-rank brick dist tests in both modes; LBM / CG N=96, 101 bitwise equal
- [x] N=512 brick 2x2x2: LBM +8.3 %, CG -2.7 %; five-panel sweep plot (slab, pencil, pencil
      Direct, brick, brick Direct)
- [x] Step tables in device memory (`PackTable`, `PackTableImage`, `pack_tables_`): one launch
      per phase whatever the step count; A/B bitwise equal, MLUPS within 0.7 %
- [x] Warp-ballot step search (`pack_prefix_count`) + one-step shortcut: brick Direct LBM gather
      76.2 -> 74.2 us (by value 70.7), Staged unchanged

### Open

- [ ] Commit (library + tests; not the /tmp scripts)
- [ ] Without `--hp-halo` the halo is not overlapped: the LBM interior fills every SM and the
      pack / NCCL kernels only start in its last ~70 us. With `--hp-halo` Staged and Direct
      converge. Consider hp-halo by default, or a smaller interior grid that leaves SMs free
- [ ] Device step table still costs ~3.5 us on the brick Direct LBM gather (74.2 vs 70.7 us by
      value): one load for the search, then the step. Only a by-value fast path for small tables
      would remove it
- [ ] Measure across nodes (edge messages are 4 KB: latency-bound off-node)
- [ ] Known unrelated failure: 4-rank `pencil-periodicX/c1/runs` (`pair_count`, run-based path)

## Halo exchange sweeps (LBM + CG)

Same matrix for both: `dense_distributed` slab, pencil (staged / direct), brick (staged /
direct) at 4 and 8 GPUs (brick 8), N = 256 ... 512, hp-halo off / on, overlap none / standard;
120 runs each plus 5 one-GPU Dense references.

### Done

- [x] LBM sweep: `overlaps` matrix key (`-ov<mode>` labels), `direct_speedup` figure, experiment
      renamed `pencil_halo_exchange` -> `halo_exchange` (+ `halo_exchange_reference`)
- [x] `cg_poisson`: `--hp-halo`, `--reps`, `--report` (`carbon-cg-report-v1`), `--label`;
      all ranks exit 0 at `--max_iter`
- [x] `benchmarks/cg/scripts/`: `cg-sweep.yml`, `cg_sweep_config.py`, `cg-sweep-run.sh`,
      `plot_cg_scaling.py` (`vs_gpus` MDOF/s per GPU, `direct_speedup`)
- [x] Smoke sweeps (N = 128 / 192, 8 GPUs, 32 runs each) and plots

### Open

- [ ] Run both full sweeps and look at the figures

## Kernel graphs (composite kernels expanded by the Skeleton)

A kernel can describe itself as a graph of child kernels with a dummy head and tail
(`detail::KernelGraph`, `KernelImpl::expand(self)`). After dependency analysis and the overlap
rewrite, `Skeleton::compile_expand_kernel_graphs_` replaces it by its children: the composite's
predecessors feed the head's successors, the tail's predecessors feed the composite's successors.

### Done

- [x] `KernelGraph` + `KernelImpl::expand` / `communicates`; `HaloKernelImplBase` forwards to
      `HaloT::kernel_graph`
- [x] Skeleton expansion after OCC: dangling children tied to head/tail, cycle and token-coverage
      checks, nested composites, empty graph, `effective_parent()`, `edges()` / `min_edges()` /
      `kernel_streams()` sized to the children, comm-order chain of communicating children
- [x] `Options::expand_kernel_graphs{true}` + `CARBON_EXPAND_KERNEL_GRAPHS=0`; both in the rank hash
- [x] `halo_messages()` lists only communicating kernels (`<halo>/<axis>.exchange`)
- [x] `HaloDistributed` packed path: `<halo>/<x|y|z|direct>.<gather|exchange|scatter>` per pass
      (`launch_phase`, `dist_messages_of_pass`, `expandable_`)
- [x] `test_skeleton_kernel_graph` (chain None/Standard, diamond + dangling, empty, nested,
      cycle / coverage rejects, opt-out, Extended)
- [x] GPU 8 ranks, N=96: LBM pencil Staged/Direct, slab, brick and CG pencil/slab, overlap
      none/standard: checksums / residuals bitwise equal on vs off; transfers identical apart from
      `seq` and labels
- [x] N=256 pencil: on vs off within noise (LBM, CG)

### Open

- [ ] Commit (library + tests; not the /tmp scripts)
- [ ] Per-face Boundary dependencies (Boundary of the Y faces waits only on the Y scatter) — the
      concurrency win this enables
- [ ] Public `Kernel::graph(...)` builder for user composites
- [ ] Expand the other halo families (intra-process multi-GPU `Halo`, Disg, mGrid)
- [ ] Expansion of composites split by OCC (Internal / Boundary views) is skipped with an info log

## Other

- [ ] Stale files reintroduced by #86 (not built by CMake): `benchmarks/lbm_lid_driven_cavity_distributed/`,
      `benchmarks/lbm_lid_driven_cavity/scripts/lbm_sweep_config.py`, `dense_distributed_window.patch`
- [ ] Commit the `carbon` submodule pointer in the outer `ai-repos` repo
