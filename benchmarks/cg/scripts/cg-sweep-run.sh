#!/usr/bin/env bash
#
# benchmarks/cg/scripts/cg-sweep-run.sh
#
# Runs the experiments of a CG sweep YAML (see cg-sweep.yml): builds
# carbon_benchmark_cg_poisson (CUDA+MPI, in Docker or on the host with
# runner: local), then does every run the selected experiments need, once.
# The CG counterpart of benchmarks/lbm_ldc/scripts/lbm-sweep-run.sh.
#
# Usage:
#   ./benchmarks/cg/scripts/cg-sweep-run.sh CONFIG.yml [--only exp1,exp2]
#
#   --only a,b   run only the named experiments of CONFIG.yml
#   DRY_RUN=1    print the resolved config, the experiments and every run, then exit
#   Env vars named after the global keys (RUNNER, FORCE, SKIP_BUILD, SM_ARCH,
#   WARMUP, ITERS_PER_REP, REPS, RUN_TIMEOUT, DATA_DIR, ...) override the YAML.
#
# dense / disg run in one process on GPUs 0..G-1; dense_distributed /
# disg_distributed run under mpiexec -n G, one GPU per rank, halos over NCCL.
# Every run does --max_iter = warmup + reps * iters_per_rep iterations with
# tolerances 1e-30 (never converges early) and writes
# <data_dir>/json/cg-<label>-n<N>-g<G>.json, <label> from cg_sweep_config.label.
# With force: false an existing report is reused only if its max_iter, warmup,
# reps and overlap match the run.
#
# Progress: <data_dir>/trace.log (one line per run, tail -f it). Failed or hung
# runs (run_timeout) are listed in <data_dir>/failed.txt and the sweep moves on.

set -euo pipefail

usage() {
  echo "usage: $0 CONFIG.yml [--only exp1,exp2]" >&2
  exit 2
}
config_path=""
only=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --only)
      [[ $# -ge 2 ]] || usage
      only="$2"
      shift 2
      ;;
    --only=*)
      only="${1#--only=}"
      shift
      ;;
    -*) usage ;;
    *)
      [[ -z "${config_path}" ]] || usage
      config_path="$1"
      shift
      ;;
  esac
done
[[ -n "${config_path}" ]] || usage
if [[ ! -f "${config_path}" ]]; then
  echo "error: config file not found: ${config_path}" >&2
  exit 2
fi
config_path="$(realpath "${config_path}")"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"

config_vars="$(CG_SWEEP_CONFIG="$(<"${config_path}")" CG_SWEEP_ONLY="${only}" \
  python3 "${repo_root}/benchmarks/cg/scripts/cg_sweep_config.py")"
while IFS='=' read -r key value; do
  [[ -z "${key}" ]] && continue
  export "${key}=${value}"
done <<< "${config_vars}"

if [[ "${SM_ARCH}" == "auto" ]]; then
  SM_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null |
             tr -d '. ' | sort -u | paste -sd';' -)" || true
  if [[ -z "${SM_ARCH}" ]]; then
    echo "error: could not detect the GPU compute capability with nvidia-smi;" \
         "set sm_arch / SM_ARCH explicitly (A100=80, H100=90)" >&2
    exit 1
  fi
fi
if [[ ! "${SM_ARCH}" =~ ^[0-9]+(\;[0-9]+)*$ ]]; then
  echo "error: SM_ARCH must be auto or numbers separated by ';' (got '${SM_ARCH}')" >&2
  exit 1
fi
export SM_ARCH

read -r -a runs <<< "${RUNS}"
read -r -a experiments <<< "${EXPERIMENTS}"
json_dir="${repo_root}/${DATA_DIR}/json"
n_existing=0
for r in "${runs[@]}"; do
  IFS=: read -r grid decomp hp ov g n label <<< "${r}"
  if [[ -f "${json_dir}/cg-${label}-n${n}-g${g}.json" ]]; then
    n_existing=$((n_existing + 1))
  fi
done

echo "==> sweep config (${config_path##*/}):"
for v in RUNNER WARMUP ITERS_PER_REP REPS OVERLAP REFERENCE_LABEL DATA_DIR SM_ARCH \
         IMAGE BUILD_DIR SKIP_BUILD FORCE RUN_TIMEOUT; do
  printf '    %-16s %s\n' "${v}" "${!v}"
done
echo "==> experiments:"
names=()
for e in "${experiments[@]}"; do
  IFS=: read -r name etype count <<< "${e}"
  names+=("${name}")
  printf '    %-24s %-15s %6d runs\n' "${name}" "${etype}" "${count}"
done
echo "==> ${#runs[@]} distinct runs; ${n_existing} reports already in ${DATA_DIR}/json"
if [[ "${DRY_RUN:-0}" -eq 1 ]]; then
  for r in "${runs[@]}"; do
    IFS=: read -r grid decomp hp ov g n label <<< "${r}"
    printf '    %-9s %s\n' "${ov}" "cg-${label}-n${n}-g${g}"
  done
  exit 0
fi

if [[ "${RUNNER}" == "docker" ]]; then
  src="/src"
else
  src="${repo_root}"
fi
export SRC="${src}"
export B_DIR="${src}/${BUILD_DIR}"
export OUT_DIR="${src}/${DATA_DIR}/json"
export DATA_OUT="${src}/${DATA_DIR}"

# in_env BODY: runs BODY with bash, in IMAGE (runner docker) or on the host
# (runner local), with the variables above.
in_env() {
  local body="$1"
  if [[ "${RUNNER}" == "docker" ]]; then
    local tz=()
    [[ -e /etc/localtime ]] && tz=(-v /etc/localtime:/etc/localtime:ro)
    docker run --rm --gpus all --ipc=host "${tz[@]}" \
      -v "${repo_root}:/src:rw" -w /src \
      -e SRC -e B_DIR -e OUT_DIR -e DATA_OUT -e SM_ARCH -e RUNS -e WARMUP \
      -e ITERS_PER_REP -e REPS -e FORCE -e RUN_TIMEOUT -e RUN_TOTAL -e SWEEP_START \
      "${IMAGE}" /bin/bash -lc "${body}"
  else
    bash -c "${body}"
  fi
}

read -r -d '' build_body <<'EOF' || true
set -euo pipefail
export PATH="${HOME:-/root}/.local/bin:${PATH}"
gen=()
if command -v ninja > /dev/null; then
  gen=(-G Ninja)
fi
echo '=== Building CUDA+MPI ==='
cmake -S "${SRC}" -B "${B_DIR}" "${gen[@]}" -DCMAKE_BUILD_TYPE=Release \
  -DCARBON_BACKEND=Cuda "-DCMAKE_CUDA_ARCHITECTURES=${SM_ARCH}" -DCARBON_ENABLE_MPI=ON
cmake --build "${B_DIR}" -j"$(nproc)" --target carbon_benchmark_cg_poisson
echo '=== Build complete ==='
EOF

if [[ "${SKIP_BUILD}" -eq 0 ]]; then
  echo "==> building ${BUILD_DIR}/ (SM_${SM_ARCH}, runner ${RUNNER})"
  in_env "${build_body}"
else
  echo "==> skip_build: reusing ${BUILD_DIR}/"
fi

read -r -d '' sweep_body <<'EOF' || true
set -uo pipefail
bin="${B_DIR}/benchmarks/cg/carbon_benchmark_cg_poisson"
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export NCCL_IB_DISABLE=1
trace="${DATA_OUT}/trace.log"
failed_log="${DATA_OUT}/failed.txt"
done_n=0 exec_n=0 exec_ms=0 n_ok=0 n_skip=0 n_redo=0 n_fail=0

fmt_dur() {
  local s="$1"
  if (( s >= 3600 )); then
    printf '%dh%02dm' $((s / 3600)) $((s % 3600 / 60))
  else
    printf '%dm%02ds' $((s / 60)) $((s % 60))
  fi
}
trace_line() {
  local width=30 fill pct elapsed eta="ETA --"
  fill=$((done_n * width / RUN_TOTAL))
  pct=$((done_n * 100 / RUN_TOTAL))
  elapsed=$(( $(date +%s) - SWEEP_START ))
  if (( exec_n > 0 )); then
    eta="ETA $(fmt_dur $(( exec_ms / exec_n * (RUN_TOTAL - done_n) / 1000 )))"
  fi
  printf '%s [%s%s] %*d/%d %3d%%  elapsed %s  %-11s | %-26s | %s\n' \
    "$(date '+%F %T')" "$(printf '%*s' "${fill}" '' | tr ' ' '#')" \
    "$(printf '%*s' $((width - fill)) '' | tr ' ' '.')" "${#RUN_TOTAL}" "${done_n}" \
    "${RUN_TOTAL}" "${pct}" "$(fmt_dur "${elapsed}")" "${eta}" "$1" "$2" >> "${trace}"
}

if [[ ! -x "${bin}" ]]; then
  echo "error: ${bin} not built (run without skip_build)" >&2
  echo "$(date '+%F %T') error: ${bin} not built" >> "${trace}"
  exit 1
fi
mkdir -p "${OUT_DIR}"
: > "${failed_log}"
iters=$((WARMUP + REPS * ITERS_PER_REP))
settings_match() {
  grep -Eq "\"max_iter\": *${iters}," "$1" && grep -Eq "\"warmup\": *${WARMUP}," "$1" &&
    grep -Eq "\"reps\": *${REPS}," "$1" && grep -Eq "\"overlap\": *\"$2\"" "$1"
}
run_log="$(mktemp)"

for r in ${RUNS}; do
  IFS=: read -r grid decomp hp ov g n label <<< "${r}"
  name="cg-${label}-n${n}-g${g}"
  report="${OUT_DIR}/${name}.json"
  redo=0
  if [[ -f "${report}" && "${FORCE}" -eq 0 ]]; then
    if settings_match "${report}" "${ov}"; then
      done_n=$((done_n + 1))
      n_skip=$((n_skip + 1))
      echo "==> [skip exists] ${name}.json"
      trace_line "skip (settings match)" "${name}"
      continue
    fi
    redo=1
  fi
  extra=()
  [[ "${hp}" == "on" ]] && extra+=(--hp-halo)
  case "${grid}" in
    *_distributed)
      cmd=(mpiexec --bind-to none -n "${g}" "${bin}")
      if [[ "${decomp}" == *-direct ]]; then
        extra+=(--decomposition "${decomp%-direct}" --halo-exchange direct)
      else
        extra+=(--decomposition "${decomp}")
      fi ;;
    *) cmd=(env CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((g - 1)))" "${bin}") ;;
  esac
  echo "==> ${name}.json$([[ "${redo}" -eq 1 ]] && echo ' (settings differ, redo)')"
  t0=$(date +%s%N)
  timeout "${RUN_TIMEOUT}" "${cmd[@]}" \
    --grid "${grid}" --nx "${n}" --ny "${n}" --nz "${n}" --max_iter "${iters}" \
    --warmup "${WARMUP}" --reps "${REPS}" --tol_rel 1e-30 --tol_abs 1e-30 \
    --report_every 1000000000 --overlap "${ov}" "${extra[@]}" \
    --label "${label}" --report "${report}" 2>&1 | tee "${run_log}"
  rc=${PIPESTATUS[0]}
  ms=$(( ($(date +%s%N) - t0) / 1000000 ))
  done_n=$((done_n + 1))
  exec_n=$((exec_n + 1))
  exec_ms=$((exec_ms + ms))
  secs="$((ms / 1000)).$((ms % 1000 / 100))s"
  if [[ "${rc}" -eq 0 && -f "${report}" ]]; then
    mdofs="$(grep -oE '^MDOFS=[0-9.]+' "${run_log}" | tail -n 1 | cut -d= -f2)"
    if [[ "${redo}" -eq 1 ]]; then
      status=redo
      n_redo=$((n_redo + 1))
    else
      status=ok
      n_ok=$((n_ok + 1))
    fi
    trace_line "$(printf '%-4s %7s MDOF/s %7s' "${status}" "${mdofs%.*}" "${secs}")" "${name}"
  else
    kind=FAIL
    [[ "${rc}" -eq 124 ]] && kind=TIMEOUT
    n_fail=$((n_fail + 1))
    echo "!!> [failed] ${name}.json"
    rm -f "${report}"
    echo "${name}.json" >> "${failed_log}"
    trace_line "$(printf '%-7s rc=%-3s %7s' "${kind}" "${rc}" "${secs}")" "${name}"
  fi
done
rm -f "${run_log}"
printf 'n_ok=%s\nn_skip=%s\nn_redo=%s\nn_fail=%s\n' "${n_ok}" "${n_skip}" "${n_redo}" \
  "${n_fail}" > "${DATA_OUT}/.trace-state"
echo "=== sweep complete (${n_fail} failed) ==="
EOF

trace="${repo_root}/${DATA_DIR}/trace.log"
state="${repo_root}/${DATA_DIR}/.trace-state"
mkdir -p "${json_dir}"
export RUN_TOTAL="${#runs[@]}"
export SWEEP_START="$(date +%s)"
rm -f "${state}"
{
  echo "======================================================================"
  echo "$(date '+%F %T') sweep start: ${config_path#"${repo_root}/"} (runner ${RUNNER})"
  echo "    experiments: ${names[*]}"
  echo "    ${RUN_TOTAL} runs, ${n_existing} reports already in ${DATA_DIR}/json, force=${FORCE}"
  echo "    timing: warmup ${WARMUP}, ${REPS} x ${ITERS_PER_REP} iterations, overlap ${OVERLAP}" \
       "(unless the experiment lists overlaps)"
} >> "${trace}"
echo "==> progress: tail -f ${DATA_DIR}/trace.log"

in_env "${sweep_body}"
if [[ "${RUNNER}" == "docker" ]]; then
  in_env "chmod -R a+rw /src/${DATA_DIR}"
fi

declare -A s=([n_ok]=0 [n_skip]=0 [n_redo]=0 [n_fail]=0)
if [[ -f "${state}" ]]; then
  while IFS='=' read -r k v; do
    [[ -n "${k}" ]] && s[${k}]="${v}"
  done < "${state}"
  rm -f "${state}"
fi
took=$(( $(date +%s) - SWEEP_START ))
{
  printf '%s sweep done: %s ok, %s skipped, %s redone, %s failed; took %dh%02dm%02ds\n' \
    "$(date '+%F %T')" "${s[n_ok]}" "${s[n_skip]}" "${s[n_redo]}" "${s[n_fail]}" \
    $((took / 3600)) $((took % 3600 / 60)) $((took % 60))
  sed 's/^/    FAILED: /' "${repo_root}/${DATA_DIR}/failed.txt" 2>/dev/null || true
} | tee -a "${trace}"
echo "==> reports in ${DATA_DIR}/json/; plot with benchmarks/cg/scripts/plot_cg_scaling.py" \
     "--config ${config_path#"${repo_root}/"}"
