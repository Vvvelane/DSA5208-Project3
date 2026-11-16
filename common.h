// Shared definitions for the serial, MPI (CPU) and MPI+CUDA Jacobi solvers.
//
// Problem: Poisson equation  laplace(u) = f  on the unit square (dim=2) or
// unit cube (dim=3) with u = 0 on the boundary. We pick the exact solution
//   u(x,y)   = sin(pi x) sin(pi y)
//   u(x,y,z) = sin(pi x) sin(pi y) sin(pi z)
// so that f = -dim * pi^2 * u and the numerical error can be measured.
//
// Grid: n interior points per dimension, spacing h = 1/(n+1). Global interior
// indices run 1..n; index 0 and n+1 are the zero boundary.
// Array layout: x fastest, then y, then z:  id = (k*(ly+2) + j)*(lx+2) + i
// In 3D the array has a halo layer in z as well; in 2D there is a single z
// layer (hz = 0).
#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

constexpr double PI = 3.14159265358979323846;

struct Params {
    int dim = 2;
    int n = 256;          // interior points per dimension
    int iters = 1000;     // maximum number of Jacobi iterations
    double tol = 0.0;     // stop when max|u_new-u_old| < tol (0 = fixed iteration count)
    int check = 100;      // evaluate the convergence test every `check` iterations
    int px = 0, py = 0, pz = 0;  // process grid (0 = use --decomp)
    int decomp = -1;      // process grid if px/py/pz are not given: 0 = 1D slab along the slowest
                          // axis, 1 = balanced (MPI_Dims_create); -1 = program default
                          // (GPU program: slab, CPU program: balanced)
    int cudaaware = 0;    // 1: pass GPU pointers to MPI, 0: stage halos through host memory
    int overlap = 0;      // 1: overlap halo exchange with the interior update
    int prio = 1;         // 1: give the boundary/communication stream a high CUDA stream priority
    std::string csv;      // append one result line to this file
};

inline Params parse_args(int argc, char** argv) {
    Params p;
    if ((argc - 1) % 2 != 0) {
        fprintf(stderr, "options come in pairs: --name value\n");
        exit(1);
    }
    for (int i = 1; i + 1 < argc; i += 2) {
        const char* key = argv[i];
        const char* val = argv[i + 1];
        if (!strcmp(key, "--dim")) p.dim = atoi(val);
        else if (!strcmp(key, "--n")) p.n = atoi(val);
        else if (!strcmp(key, "--iters")) p.iters = atoi(val);
        else if (!strcmp(key, "--tol")) p.tol = atof(val);
        else if (!strcmp(key, "--check")) p.check = atoi(val);
        else if (!strcmp(key, "--px")) p.px = atoi(val);
        else if (!strcmp(key, "--py")) p.py = atoi(val);
        else if (!strcmp(key, "--pz")) p.pz = atoi(val);
        else if (!strcmp(key, "--decomp")) p.decomp = atoi(val);
        else if (!strcmp(key, "--cudaaware")) p.cudaaware = atoi(val);
        else if (!strcmp(key, "--overlap")) p.overlap = atoi(val);
        else if (!strcmp(key, "--prio")) p.prio = atoi(val);
        else if (!strcmp(key, "--csv")) p.csv = val;
        else {
            fprintf(stderr, "unknown option %s\n", key);
            exit(1);
        }
    }
    if ((p.dim != 2 && p.dim != 3) || p.n < 1 || p.iters < 1 || p.check < 1) {
        fprintf(stderr, "invalid parameters (dim must be 2 or 3)\n");
        exit(1);
    }
    return p;
}

HD inline double exact_u(int dim, double x, double y, double z) {
    double v = sin(PI * x) * sin(PI * y);
    if (dim == 3) v *= sin(PI * z);
    return v;
}

HD inline double source_f(int dim, double x, double y, double z) {
    return -dim * PI * PI * exact_u(dim, x, y, z);
}

// Error and checksum accumulators over a set of grid points.
struct Norms {
    double sq_err = 0.0, max_err = 0.0, sum_u = 0.0;
    void add(double u, double ue) {
        double e = fabs(u - ue);
        sq_err += e * e;
        if (e > max_err) max_err = e;
        sum_u += u;
    }
};

// All times in seconds (wall clock).
struct Times {
    double total = 0, comm = 0, check = 0;  // Jacobi loop; halo exchange and convergence tests inside it
    double pack = 0, mpi = 0, unpack = 0;   // comm split: pack kernel + GPU->host copy, MPI, host->GPU copy + unpack
    double setup = 0, final_ = 0, wall = 0; // allocation + initialisation; copy-back + error norms; whole program
};

inline void report(const Params& P, const char* version, int ranks, int px, int py, int pz,
                   int iters_done, const Times& t, double diff, const Norms& nm, double h) {
    double l2 = sqrt(pow(h, P.dim) * nm.sq_err);
    printf("version=%s dim=%d n=%d ranks=%d (px,py,pz)=(%d,%d,%d) cudaaware=%d overlap=%d prio=%d\n",
           version, P.dim, P.n, ranks, px, py, pz, P.cudaaware, P.overlap, P.prio);
    printf("iters=%d loop=%.6f s (comm=%.6f [pack=%.6f mpi=%.6f unpack=%.6f] check=%.6f) "
           "final_diff=%.3e\n",
           iters_done, t.total, t.comm, t.pack, t.mpi, t.unpack, t.check, diff);
    printf("setup=%.6f s final=%.6f s wall=%.6f s\n", t.setup, t.final_, t.wall);
    printf("l2_err=%.6e max_err=%.6e sum_u=%.15e\n", l2, nm.max_err, nm.sum_u);
    if (P.csv.empty()) return;
    FILE* f = fopen(P.csv.c_str(), "a");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", P.csv.c_str());
        return;
    }
    fseek(f, 0, SEEK_END);
    if (ftell(f) == 0)
        fprintf(f, "version,dim,n,ranks,px,py,pz,cudaaware,overlap,prio,check_every,iters,total_s,"
                   "comm_s,pack_s,mpi_s,unpack_s,check_s,setup_s,final_s,wall_s,"
                   "final_diff,l2_err,max_err,sum_u\n");
    fprintf(f, "%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,"
               "%.6e,%.15e,%.15e,%.15e\n",
            version, P.dim, P.n, ranks, px, py, pz, P.cudaaware, P.overlap, P.prio, P.check,
            iters_done, t.total, t.comm, t.pack, t.mpi, t.unpack, t.check, t.setup, t.final_,
            t.wall, diff, l2, nm.max_err, nm.sum_u);
    fclose(f);
}
