# Helpers shared by run_gpu.sh and run_cpu.sh.  Source it (after setting NAME=gpu|cpu); do not run it.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
RES=${RES:-results}
MPIRUN=${MPIRUN:-mpirun}
REPS=${REPS:-3}            # repetitions of every parallel configuration (medians are used)
SREPS=${SREPS:-1}          # repetitions of the slow serial runs
BIN=${BIN:-bin_${NAME}_${RES##*/}}   # separate build dir per job: concurrent jobs must not rebuild the same files
S=${S:-$BIN/jacobi_serial}
WANT=${1:-all}             # one stage name or "all"
T_START=$(date +%s)
FAILS=0
STAGE_FAILS0=0

mkdir -p "$RES"
LOG=$RES/run_$NAME.log
ERR=$RES/stderr_$NAME.log
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# to <seconds> <command...>: run the command with a time limit (a hung run must not use up the whole
# walltime); without the `timeout` tool the command just runs without limit
RUN_TIMEOUT=${RUN_TIMEOUT:-900}
to() { if command -v timeout >/dev/null 2>&1; then timeout "$@"; else shift; "$@"; fi; }

# exec_run <csv> <np> <exe> <args...>      (np=0: run <exe> directly, no mpirun)
exec_run() {
    local csv=$1 np=$2 exe=$3; shift 3
    local cmd
    if [[ $np == 0 ]]; then cmd=("$exe" "$@" --csv "$RES/$csv")
    else cmd=($MPIRUN -np "$np" "$exe" "$@" --csv "$RES/$csv"); fi
    to "$RUN_TIMEOUT" "${cmd[@]}" >/dev/null 2>>"$ERR" || { log "FAILED: ${cmd[*]}"; FAILS=$((FAILS + 1)); }
}
# rep <times> <csv> <np> <exe> <args...>
rep() { local k=$1 _; shift; for _ in $(seq "$k"); do exec_run "$@"; done; }

# begin <stage> <csv...>: returns 0 if the stage has to run now, 1 if it is skipped
begin() {
    local name=$1; shift
    [[ $WANT == all || $WANT == "$name" ]] || return 1
    if [[ -f $RES/.done_${NAME}_$name && -z $FORCE ]]; then
        log "skip stage $name (already done; FORCE=1 to redo)"; return 1
    fi
    mkdir -p "$RES/old"
    for c in "$@"; do [[ -f $RES/$c ]] && mv "$RES/$c" "$RES/old/$(date +%m%d%H%M%S)_$c"; done
    rm -f "$RES/.done_${NAME}_$name"
    STAGE_FAILS0=$FAILS
    log "=== stage $name"
    return 0
}
# a stage only counts as done when none of its runs failed
finish() {
    if [[ $FAILS -eq $STAGE_FAILS0 ]]; then touch "$RES/.done_${NAME}_$1"
    else log "stage $1 had $((FAILS - STAGE_FAILS0)) failed runs: not marked done"; fi
    log "--- stage $1 finished ($(( $(date +%s) - T_START )) s since start)"
}

# post-processing: correctness checks and figures
post() {
    log "=== validation"
    python3 analysis/validate.py "$RES" 2>&1 | tee -a "$LOG"
    log "=== figures"
    if python3 -c "import pandas, matplotlib" 2>/dev/null; then
        python3 analysis/plot.py "$RES" 2>&1 | tee -a "$LOG"
    else
        log "pandas/matplotlib not installed here: copy $RES/ to your laptop and run: python3 analysis/plot.py $RES"
    fi
}
done_msg() { log "done in $(( $(date +%s) - T_START )) s, failed runs: $FAILS (details: $ERR)"; }
