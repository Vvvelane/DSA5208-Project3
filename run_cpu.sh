#!/bin/bash
# CPU experiments: serial size scaling and the MPI (CPU) program scaled up to 256 cores.
# Needs a multi-node job (qsub scripts/job_cpu.pbs); inside it PBS_NODEFILE lists the cores.
#
#   bash scripts/run_cpu.sh              # everything (resumable: finished stages are skipped)
#   bash scripts/run_cpu.sh strong       # one stage: validate serial strong weak strategies
#   FORCE=1 bash scripts/run_cpu.sh      # redo finished stages (old CSVs go to results/old/)
#
# Output: results/serial.csv, results/cpu_*.csv, results/run_cpu.log, then validation + figures.
NAME=cpu
# inside a PBS job start the ranks on the hosts of the job (this OpenMPI has no PBS support)
if [[ -n $PBS_NODEFILE && -z $MPIRUN ]]; then
    # --prefix: the ssh-started daemons on the other nodes must find orted and the MPI libraries
    MPIRUN="mpirun --prefix $(dirname "$(dirname "$(command -v mpirun)")") --hostfile $PBS_NODEFILE --map-by core --bind-to core"
fi
source "$(dirname "$0")/lib.sh"

# QUICK=1: tiny sizes, only to test the scripts and the cluster set-up (use RES=results_quick)
if [[ -n $QUICK ]]; then
    : "${RANKS:=1 2 4 8}" "${STRONG:=2:256:50 3:32:20}" "${WEAK:=2:64:30 3:16:20}" "${STRAT:=2:256:50 3:32:20}"
    : "${PS:=4 8}" "${SIZES2:=64 128}" "${SIZES3:=16 32}" "${ITER2:=50}" "${ITER3:=20}" "${CHKN:=255}"
    : "${CHKIT:=2000}" "${REPS:=2}"
fi
M=${M:-$BIN/jacobi_mpi}
RANKS=${RANKS:-"1 2 4 8 16 32 64 128 256"}
# strong scaling: dim:n:iterations   (fixed problem, more and more cores)
STRONG=${STRONG:-"2:2048:500 2:8192:200 3:128:500 3:512:100"}
# weak scaling: dim:n per 1 core:iterations   (n grows so that the points per core stay constant)
WEAK=${WEAK:-"2:1024:300 2:4096:200 3:64:300 3:256:100"}
# strategies at PS cores: dim:n:iterations
STRAT=${STRAT:-"2:8192:200 3:512:100"}
PS=${PS:-"64 256"}
SIZES2=${SIZES2:-"128 256 512 1024 2048 4096 8192"}; ITER2=${ITER2:-200}   # serial scaling, 2D
SIZES3=${SIZES3:-"32 64 128 256 512"}; ITER3=${ITER3:-100}                  # serial scaling, 3D
CHKN=${CHKN:-2047}; CHKIT=${CHKIT:-50000}

# ---------------------------------------------------------------- environment
if [[ -n $PBS_NODEFILE ]]; then NSLOTS=$(wc -l < "$PBS_NODEFILE"); else NSLOTS=${NSLOTS:-$(nproc)}; fi
log "cores available: $NSLOTS on $( [[ -n $PBS_NODEFILE ]] && sort -u "$PBS_NODEFILE" | wc -l || echo 1 ) node(s), mpirun: $MPIRUN"
keep() { local r out=""; for r in $RANKS; do [[ $r -le $NSLOTS ]] && out="$out $r"; done; echo $out; }
RANKS=$(keep)
log "rank counts that will be measured: $RANKS"
if [[ -z $SKIP_BUILD ]]; then
    make -B BIN="$BIN" serial mpi >>"$LOG" 2>&1 || { log "build failed (see $LOG)"; exit 1; }
fi
chmod +x "$BIN"/* 2>/dev/null
if ! to 120 $MPIRUN -np 2 "$M" --dim 2 --n 32 --iters 10 >/dev/null 2>>"$ERR"; then
    log "smoke test of $M failed (see $ERR), stopping"; exit 1
fi

# ---------------------------------------------------------------- 1. validation
if begin validate cpu_validate.csv; then
    # fixed iteration count: the MPI program must reproduce the serial result for every number of
    # ranks, process-grid shape and with/without overlap (65 and 33 give uneven splits)
    for cfg in "2 64" "2 65" "3 32" "3 33"; do
        set -- $cfg; d=$1; n=$2
        exec_run cpu_validate.csv 0 "$S" --dim $d --n $n --iters 2000
        for np in 1 2 4 6; do
            exec_run cpu_validate.csv $np "$M" --dim $d --n $n --iters 2000 --overlap 0
            exec_run cpu_validate.csv $np "$M" --dim $d --n $n --iters 2000 --overlap 1
        done
        for np in 4 6; do
            exec_run cpu_validate.csv $np "$M" --dim $d --n $n --iters 2000 --overlap 0 --decomp 0
        done
    done
    finish validate
fi

# ---------------------------------------------------------------- 2. serial runtime vs problem size
if begin serial serial.csv; then
    for n in $SIZES2; do
        r=$REPS; [[ $n -gt 2048 ]] && r=1
        rep $r serial.csv 0 "$S" --dim 2 --n $n --iters $ITER2
    done
    for n in $SIZES3; do
        r=$REPS; [[ $n -gt 128 ]] && r=1
        rep $r serial.csv 0 "$S" --dim 3 --n $n --iters $ITER3
    done
    finish serial
fi

# ---------------------------------------------------------------- 3. strong scaling
if begin strong cpu_strong.csv; then
    for w in $STRONG; do
        IFS=: read -r d n it <<<"$w"
        rep $SREPS cpu_strong.csv 0 "$S" --dim $d --n $n --iters $it
        for np in $RANKS; do rep $REPS cpu_strong.csv $np "$M" --dim $d --n $n --iters $it --overlap 0; done
    done
    finish strong
fi

# ---------------------------------------------------------------- 4. weak scaling
if begin weak cpu_weak.csv; then
    for w in $WEAK; do
        IFS=: read -r d n0 it <<<"$w"
        for np in $RANKS; do
            # same number of points per core: n^dim = n0^dim * np
            n=$(awk -v n0=$n0 -v p=$np -v d=$d 'BEGIN{printf "%d", n0 * p^(1.0/d) + 0.5}')
            rep $REPS cpu_weak.csv $np "$M" --dim $d --n $n --iters $it --overlap 0
        done
    done
    finish weak
fi

# ---------------------------------------------------------------- 5. strategies
if begin strategies cpu_strategies.csv cpu_check.csv; then
    for p in $PS; do
        [[ $p -le $NSLOTS ]] || { log "skip $p cores (only $NSLOTS available)"; continue; }
        for w in $STRAT; do
            IFS=: read -r d n it <<<"$w"
            # shape of the process grid (1D slabs vs balanced) and overlap of exchange and update
            for dc in 0 1; do
                for ov in 0 1; do
                    rep $REPS cpu_strategies.csv $p "$M" --dim $d --n $n --iters $it --overlap $ov --decomp $dc
                done
            done
        done
    done
    # how often the convergence test (MPI_Allreduce) is evaluated; tol is unreachable so every run
    # does exactly CHKIT iterations
    p=64; [[ $NSLOTS -lt 64 ]] && p=$NSLOTS
    for chk in 1 10 100; do
        rep $REPS cpu_check.csv $p "$M" --dim 2 --n $CHKN --iters $CHKIT --tol 1e-30 --check $chk --overlap 0
    done
    finish strategies
fi

[[ $WANT == all ]] && post
done_msg
