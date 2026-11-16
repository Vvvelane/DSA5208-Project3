# Jacobi iteration for the 2D / 3D Poisson equation: serial, MPI (CPU) and MPI+CUDA

Solves `laplace(u) = f` on the unit square / cube with `u = 0` on the boundary.
The exact solution `sin(pi x) sin(pi y) [sin(pi z)]` fixes `f`, so the error can be measured.

| file | content |
|---|---|
| `src/common.h` | parameters, exact solution, error norms, CSV output |
| `src/jacobi_serial.cpp` | CPU baseline (2D and 3D) |
| `src/jacobi_mpi.cpp` | MPI program on CPU cores, one rank per core |
| `src/jacobi_mpi_cuda.cu` | MPI+CUDA program, one rank per GPU (host-staged or CUDA-aware halo exchange) |
| `scripts/run_gpu.sh`, `scripts/job_gpu.pbs` | all MPI+CUDA experiments (2 GPUs), as a batch job |
| `scripts/run_cpu.sh`, `scripts/job_cpu.pbs` | serial size scaling and MPI scaling up to 256 cores, as a batch job |
| `scripts/lib.sh` | helper functions shared by the two run scripts |
| `analysis/validate.py`, `analysis/plot.py` | correctness checks and figures |

## Build (on Vanda)

```bash
# GPU node: CUDA-aware MPI (recommended module) and the CUDA compiler
module load nvhpc-hpcx-cuda12/26.1
make serial gpu     # bin/jacobi_serial and bin/jacobi_mpi_cuda   (CUDA_ARCH=sm_86 for the A40)
make gpu-alt        # fallback if "make gpu" fails: g++ as host compiler for nvcc

# CPU nodes
module load openmpi4/gcc/4.1.8
make serial mpi     # bin/jacobi_serial and bin/jacobi_mpi
```

Check whether the MPI is CUDA-aware: `ompi_info --parsable --all | grep mpi_built_with_cuda_support:value`.

## Run one case

```bash
bin/jacobi_serial --dim 2 --n 1024 --iters 1000
mpirun -np 64 bin/jacobi_mpi --dim 3 --n 256 --iters 100 --decomp 1 --overlap 1
mpirun -np 2 bin/jacobi_mpi_cuda --dim 3 --n 256 --iters 500 --cudaaware 1 --overlap 1
```

Options (all programs): `--dim 2|3`, `--n` interior points per dimension, `--iters` (maximum) iterations,
`--tol` stop when max change < tol (0 = fixed iteration count), `--check` convergence-test interval,
`--csv file` (append one result line).
Parallel programs: `--px --py --pz` process grid, or `--decomp 0|1` (0 = 1D slabs along the slowest axis,
1 = balanced grid from `MPI_Dims_create`; default: balanced for `jacobi_mpi`, slabs for `jacobi_mpi_cuda`),
`--overlap 0|1` (overlap the halo exchange with the update of the interior).
MPI+CUDA only: `--cudaaware 0|1` (0 = stage halos through pinned host memory, 1 = pass GPU pointers to MPI),
`--prio 0|1` (high priority for the CUDA stream of the boundary update and the communication).

## Run all experiments (two independent batch jobs)

Submit from the project directory (these are the resources the scripts request):

```bash
qsub scripts/job_gpu.pbs     # select=2:ngpus=1, place=pack: stages validate, accuracy, strong, strategies, weak
qsub scripts/job_cpu.pbs     # many cores over several nodes: stages validate, serial, strong, weak, strategies
```

Each job first builds the programs and runs the validation stage, stops if a check fails, and then runs
the remaining stages. Finished stages are skipped when a script is started again
(`FORCE=1` repeats them); one stage can be run alone, e.g. `bash scripts/run_gpu.sh strong`.
Progress: `results/run_gpu.log`, `results/run_cpu.log`; errors of single runs: `results/stderr_*.log`.
Problem sizes and repetitions can be changed with environment variables (see the top of the run scripts).

Outputs in `results/`: `gpu_*.csv`, `cpu_*.csv`, `serial.csv`, `accuracy.csv` (one line per run:
loop time, communication split into pack / MPI / unpack, convergence-check time, setup and final-evaluation
time, wall time, errors).

## Validation and figures

```bash
python3 analysis/validate.py results    # parallel == serial, error shrinks 4x per grid refinement
python3 analysis/plot.py results        # figures (*.png) and summary tables (summary_*.csv)
```

`plot.py` needs numpy, pandas and matplotlib; if they are missing on the cluster, copy `results/` to a
computer that has them. Time, problem size and core count are drawn on logarithmic axes.
