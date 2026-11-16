#!/bin/bash
# MPI+CUDA experiments: needs a node with 2 GPUs (qsub scripts/job_gpu.pbs, or an interactive session).
#
#   bash scripts/run_gpu.sh              # everything (resumable: finished stages are skipped)
#   bash scripts/run_gpu.sh strong       # one stage: validate accuracy strong strategies weak
#   FORCE=1 bash scripts/run_gpu.sh      # redo finished stages (old CSVs go to results/old/)
#
# Output: results/gpu_*.csv, results/accuracy.csv, results/run_gpu.log, then validation + figures.
# Sizes and repetitions can be overridden from the environment (see the defaults below).
NAME=gpu
source "$(dirname "$0")/lib.sh"

# QUICK=1: tiny sizes, only to test the scripts and the cluster set-up (use RES=results_quick)
if [[ -n $QUICK ]]; then
    : "${SIZES2:=256 512}" "${SIZES3:=32 64}" "${ITER2:=100}" "${ITER3:=50}" "${N2:=512}" "${N3:=64}"
    : "${N2S:=256}" "${N3S:=32}" "${ITERS:=200}" "${WEAK:=2:256:362 3:32:40}" "${CHKN:=255}" "${CHKIT:=2000}"
    : "${REPS:=2}"
fi
G=${G:-$BIN/jacobi_mpi_cuda}
ITER2=${ITER2:-1000}; ITER3=${ITER3:-500}
SIZES2=${SIZES2:-"1024 2048 4096 8192"}   # strong scaling, 2D
SIZES3=${SIZES3:-"128 256 512"}           # strong scaling, 3D
N2=${N2:-8192}; N3=${N3:-512}             # large problems of the strategy study
N2S=${N2S:-1024}; N3S=${N3S:-128}; ITERS=${ITERS:-5000}   # small problems (communication matters)
WEAK=${WEAK:-"2:2048:2896 2:8192:11585 3:256:323 3:512:645"}  # dim:n(1 GPU):n(2 GPUs)
CHKN=${CHKN:-2047}; CHKIT=${CHKIT:-100000}

# ---------------------------------------------------------------- environment
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
log "GPUs visible: $NGPU, mpirun: $(command -v ${MPIRUN%% *})"
if [[ $NGPU -lt 2 && -z $ALLOW_SHARED ]]; then
    log "this script needs 2 GPUs (ALLOW_SHARED=1 lets 2 ranks share one GPU: correctness only, no scaling data)"
    exit 1
fi
if [[ -z $SKIP_BUILD ]]; then
    # -B: always rebuild (an old binary copied over from elsewhere may look up to date)
    if ! make -B BIN="$BIN" serial gpu >>"$LOG" 2>&1; then
        log "make gpu failed, trying the fallback (g++ as host compiler)"
        make -B BIN="$BIN" serial gpu-alt >>"$LOG" 2>&1 || { log "build failed (see $LOG)"; exit 1; }
    fi
fi
chmod +x "$BIN"/* 2>/dev/null
# smoke tests: stop right here instead of producing a pile of failed runs
if ! to 120 $MPIRUN -np 2 "$G" --dim 2 --n 32 --iters 10 --cudaaware 0 >/dev/null 2>>"$ERR"; then
    log "smoke test of $G failed (see $ERR), stopping"; exit 1
fi
# CUDA-aware MPI: use it only if it really works (module nvhpc-hpcx-cuda12/26.1)
CAS="0"
if [[ -z $NO_CUDAAWARE ]] && to 120 $MPIRUN -np 2 "$G" --dim 2 --n 32 --iters 10 --cudaaware 1 >/dev/null 2>>"$ERR"; then
    CAS="0 1"
fi
log "cudaaware settings that will be measured: $CAS"
ca0=${CAS%% *}   # 0 = host-staged halos (always measured)

# ---------------------------------------------------------------- 1. validation
if begin validate gpu_validate.csv; then
    # fixed iteration count: every parallel variant must reproduce the serial result
    # (65 and 33 give uneven splits)
    for cfg in "2 64" "2 65" "3 32" "3 33"; do
        set -- $cfg; d=$1; n=$2
        exec_run gpu_validate.csv 0 "$S" --dim $d --n $n --iters 2000
        for ca in $CAS; do
            exec_run gpu_validate.csv 1 "$G" --dim $d --n $n --iters 2000 --overlap 0 --cudaaware $ca
            exec_run gpu_validate.csv 2 "$G" --dim $d --n $n --iters 2000 --overlap 0 --cudaaware $ca
            exec_run gpu_validate.csv 2 "$G" --dim $d --n $n --iters 2000 --overlap 1 --prio 0 --cudaaware $ca
            exec_run gpu_validate.csv 2 "$G" --dim $d --n $n --iters 2000 --overlap 1 --prio 1 --cudaaware $ca
        done
        if [[ $d == 2 ]]; then cuts=("2 1 1" "1 2 1"); else cuts=("2 1 1" "1 2 1" "1 1 2"); fi
        for c in "${cuts[@]}"; do
            set -- $c
            exec_run gpu_validate.csv 2 "$G" --dim $d --n $n --iters 2000 --overlap 0 --cudaaware $ca0 \
                --px $1 --py $2 --pz $3
        done
    done
    finish validate
fi

# ---------------------------------------------------------------- 2. accuracy
if begin accuracy accuracy.csv; then
    # converged runs: the error against the exact solution drops 4x per grid refinement,
    # and the iteration count grows like n^2
    for n in 15 31 63 127; do
        exec_run accuracy.csv 1 "$G" --dim 2 --n $n --iters 400000 --tol 1e-12 --check 100 --cudaaware $ca0
    done
    for n in 15 31 63; do
        exec_run accuracy.csv 1 "$G" --dim 3 --n $n --iters 400000 --tol 1e-12 --check 100 --cudaaware $ca0
    done
    finish accuracy
fi

# ---------------------------------------------------------------- 3. strong scaling (1 vs 2 GPUs)
if begin strong gpu_strong.csv; then
    for d in 2 3; do
        if [[ $d == 2 ]]; then sizes=$SIZES2; it=$ITER2; else sizes=$SIZES3; it=$ITER3; fi
        for n in $sizes; do
            rep $SREPS gpu_strong.csv 0 "$S" --dim $d --n $n --iters $it
            rep $REPS gpu_strong.csv 1 "$G" --dim $d --n $n --iters $it --overlap 0 --cudaaware $ca0
            for ca in $CAS; do
                rep $REPS gpu_strong.csv 2 "$G" --dim $d --n $n --iters $it --overlap 0 --cudaaware $ca
            done
        done
    done
    finish strong
fi

# ---------------------------------------------------------------- 4. strategies (2 GPUs)
if begin strategies gpu_strategies.csv gpu_check.csv; then
    for d in 2 3; do
        if [[ $d == 2 ]]; then n=$N2; it=$ITER2; ns=$N2S; cuts=("2 1 1" "1 2 1")
        else n=$N3; it=$ITER3; ns=$N3S; cuts=("2 1 1" "1 2 1" "1 1 2"); fi
        # (a) halo exchange through host memory vs CUDA-aware MPI, with and without overlap of the
        #     exchange and the interior update (prio: stream priority of the boundary/comm stream)
        for sz in "$n $it" "$ns $ITERS"; do
            set -- $sz
            for ca in $CAS; do
                rep $REPS gpu_strategies.csv 2 "$G" --dim $d --n $1 --iters $2 --overlap 0 --cudaaware $ca
                rep $REPS gpu_strategies.csv 2 "$G" --dim $d --n $1 --iters $2 --overlap 1 --prio 0 --cudaaware $ca
                rep $REPS gpu_strategies.csv 2 "$G" --dim $d --n $1 --iters $2 --overlap 1 --prio 1 --cudaaware $ca
            done
        done
        # (b) which axis is cut (no overlap isolates the cost of the exchange)
        for sz in "$n $it" "$ns $ITERS"; do
            for c in "${cuts[@]}"; do
                set -- $sz $c
                rep $REPS gpu_strategies.csv 2 "$G" --dim $d --n $1 --iters $2 --overlap 0 \
                    --cudaaware $ca0 --px $3 --py $4 --pz $5
            done
        done
    done
    # (c) how often the convergence test (GPU reduction + MPI_Allreduce) is evaluated;
    #     tol is unreachable so every run does exactly CHKIT iterations
    for chk in 1 10 100; do
        rep $REPS gpu_check.csv 2 "$G" --dim 2 --n $CHKN --iters $CHKIT --tol 1e-30 --check $chk \
            --overlap 0 --cudaaware $ca0
    done
    finish strategies
fi

# ---------------------------------------------------------------- 5. weak scaling
if begin weak gpu_weak.csv; then
    # same work per GPU: the grid of 2 GPUs has 2x the points of the 1-GPU grid
    for w in $WEAK; do
        IFS=: read -r d n1 n2 <<<"$w"
        it=$ITER2; [[ $d == 3 ]] && it=$ITER3
        rep $REPS gpu_weak.csv 1 "$G" --dim $d --n $n1 --iters $it --overlap 0 --cudaaware $ca0
        for ca in $CAS; do
            rep $REPS gpu_weak.csv 2 "$G" --dim $d --n $n2 --iters $it --overlap 0 --cudaaware $ca
        done
    done
    finish weak
fi

[[ $WANT == all ]] && post
done_msg
