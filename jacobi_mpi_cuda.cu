// MPI+CUDA Jacobi iteration for the 2D / 3D Poisson equation.
//
// One MPI rank drives one GPU. The global grid is split over a Cartesian
// process grid (px,py,pz); every rank holds its block plus a one-cell halo.
// Strategies that can be switched from the command line:
//   --px/--py/--pz  process-grid shape (which axes are cut); --decomp 0 slab (default), 1 balanced
//   --cudaaware     0: stage halos through pinned host memory, 1: pass GPU pointers to MPI
//   --overlap       0: update everything, then exchange; 1: update the cells next to the
//                   faces first, exchange while the interior is being updated
//   --check         how often the convergence test (GPU reduction + MPI_Allreduce) runs
#include <mpi.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include "common.h"

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t e_ = (call);                                                          \
        if (e_ != cudaSuccess) {                                                          \
            fprintf(stderr, "CUDA error '%s' at %s:%d\n", cudaGetErrorString(e_), __FILE__, \
                    __LINE__);                                                            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                 \
        }                                                                                 \
    } while (0)

struct Grid {
    int dim, lx, ly, lz, hz;  // local interior size and z-halo width (0 in 2D, 1 in 3D)
    int offx, offy, offz;     // number of global interior points before this block
    double h;
};

struct Face {
    int dir, side, nb, count, pack_plane, unpack_plane, stag, rtag;
    size_t bytes;
    double *d_send, *d_recv, *h_send, *h_recv;
};

__device__ __forceinline__ size_t at(const Grid& g, int i, int j, int k) {
    return ((size_t)k * (g.ly + 2) + j) * (g.lx + 2) + i;
}

__global__ void init_f(double* f, Grid g) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + g.hz;
    if (i > g.lx || j > g.ly || k >= g.hz + g.lz) return;
    double x = (g.offx + i) * g.h, y = (g.offy + j) * g.h;
    double z = (g.dim == 3) ? (g.offz + k) * g.h : 0.0;
    f[at(g, i, j, k)] = source_f(g.dim, x, y, z);
}

// mode 0: all cells, 1: only cells next to a face (needed by neighbours), 2: the rest
__global__ void jacobi_kernel(const double* __restrict__ u, double* __restrict__ un,
                              const double* __restrict__ f, Grid g, int mode) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + g.hz;
    if (i > g.lx || j > g.ly || k >= g.hz + g.lz) return;
    if (mode != 0) {
        bool bnd = (i == 1 || i == g.lx || j == 1 || j == g.ly);
        if (g.dim == 3) bnd = bnd || k == g.hz || k == g.hz + g.lz - 1;
        if ((mode == 1) != bnd) return;
    }
    size_t sy = g.lx + 2, sz = sy * (g.ly + 2);
    size_t id = at(g, i, j, k);
    double s = u[id - 1] + u[id + 1] + u[id - sy] + u[id + sy];
    if (g.dim == 3) s += u[id - sz] + u[id + sz];
    un[id] = (s - g.h * g.h * f[id]) / (2.0 * g.dim);
}

// Copy one face (or halo plane) of u to/from a contiguous buffer.
__global__ void face_kernel(double* u, Grid g, int dir, int plane, double* buf, int unpack,
                            int count) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count) return;
    int i, j, k;
    if (dir == 0) {
        j = t % g.ly + 1; k = t / g.ly + g.hz; i = plane;
    } else if (dir == 1) {
        i = t % g.lx + 1; k = t / g.lx + g.hz; j = plane;
    } else {
        i = t % g.lx + 1; j = t / g.lx + 1; k = plane;
    }
    size_t id = at(g, i, j, k);
    if (unpack) u[id] = buf[t];
    else buf[t] = u[id];
}

// Per-block max of |a-b| over the interior.
__global__ void maxdiff_kernel(const double* a, const double* b, Grid g, double* partial) {
    __shared__ double sh[256];
    double m = 0.0;
    size_t cells = (size_t)g.lx * g.ly * g.lz;
    for (size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x; c < cells;
         c += (size_t)gridDim.x * blockDim.x) {
        int i = c % g.lx + 1;
        int j = (c / g.lx) % g.ly + 1;
        int k = c / ((size_t)g.lx * g.ly) + g.hz;
        size_t id = at(g, i, j, k);
        m = fmax(m, fabs(a[id] - b[id]));
    }
    sh[threadIdx.x] = m;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] = fmax(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[blockIdx.x] = sh[0];
}

static void split(int n, int p, int c, int& len, int& off) {
    int q = n / p, r = n % p;
    len = q + (c < r ? 1 : 0);
    off = c * q + std::min(c, r);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    double t_prog0 = MPI_Wtime();
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    Params P = parse_args(argc, argv);
    const int d = P.dim, n = P.n;

    // ---- process grid: default is a 1D slab cut along the slowest axis
    int dims[3] = {P.px, P.py, P.pz};
    if (!dims[0] && !dims[1] && !dims[2]) {
        if (P.decomp == 1) {  // balanced process grid
            dims[0] = dims[1] = dims[2] = 0;
            if (d == 2) dims[2] = 1;
            MPI_Dims_create(size, 3, dims);
        } else {              // default: 1D slab along the slowest axis
            dims[0] = dims[1] = dims[2] = 1;
            dims[d - 1] = size;
        }
    }
    for (int a = 0; a < 3; ++a)
        if (!dims[a]) dims[a] = 1;
    if (dims[0] * dims[1] * dims[2] != size || (d == 2 && dims[2] != 1)) {
        if (rank == 0) fprintf(stderr, "process grid does not match %d ranks / dim %d\n", size, d);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int periods[3] = {0, 0, 0}, coords[3];
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &cart);
    MPI_Comm_rank(cart, &rank);
    MPI_Cart_coords(cart, rank, 3, coords);

    Grid g;
    g.dim = d;
    g.hz = (d == 3) ? 1 : 0;
    g.h = 1.0 / (n + 1);
    split(n, dims[0], coords[0], g.lx, g.offx);
    split(n, dims[1], coords[1], g.ly, g.offy);
    if (d == 3) split(n, dims[2], coords[2], g.lz, g.offz);
    else { g.lz = 1; g.offz = 0; }
    if (g.lx < 1 || g.ly < 1 || g.lz < 1) {
        fprintf(stderr, "grid too small for this process grid\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // ---- one GPU per rank (local rank on the node)
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
    int lrank, ndev;
    MPI_Comm_rank(node, &lrank);
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    CUDA_CHECK(cudaSetDevice(lrank % ndev));

    const size_t nelem = (size_t)(g.lx + 2) * (g.ly + 2) * (g.lz + 2 * g.hz);
    double *d_u, *d_un, *d_f;
    CUDA_CHECK(cudaMalloc(&d_u, nelem * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_un, nelem * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_f, nelem * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_u, 0, nelem * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_un, 0, nelem * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_f, 0, nelem * sizeof(double)));

    dim3 blk = (d == 2) ? dim3(32, 8, 1) : dim3(32, 4, 2);
    dim3 grid((g.lx + blk.x - 1) / blk.x, (g.ly + blk.y - 1) / blk.y,
              (g.lz + blk.z - 1) / blk.z);
    init_f<<<grid, blk>>>(d_f, g);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- halo faces: direction dir in {x,y,z}, side 0 = low neighbour, 1 = high neighbour
    std::vector<Face> faces;
    for (int dir = 0; dir < d; ++dir) {
        int lo, hi;
        MPI_Cart_shift(cart, dir, 1, &lo, &hi);
        for (int side = 0; side < 2; ++side) {
            Face f;
            f.dir = dir;
            f.side = side;
            f.nb = side ? hi : lo;
            if (f.nb == MPI_PROC_NULL) continue;
            int len = dir == 0 ? g.lx : dir == 1 ? g.ly : g.lz;
            f.count = dir == 0 ? g.ly * g.lz : dir == 1 ? g.lx * g.lz : g.lx * g.ly;
            f.pack_plane = side ? (dir == 2 ? g.hz + g.lz - 1 : len) : (dir == 2 ? g.hz : 1);
            f.unpack_plane = side ? (dir == 2 ? g.hz + g.lz : len + 1) : (dir == 2 ? g.hz - 1 : 0);
            f.stag = 2 * dir + side;        // tag of the face I send
            f.rtag = 2 * dir + (1 - side);  // tag of the face my neighbour sends into this halo
            f.bytes = (size_t)f.count * sizeof(double);
            CUDA_CHECK(cudaMalloc(&f.d_send, f.bytes));
            CUDA_CHECK(cudaMalloc(&f.d_recv, f.bytes));
            f.h_send = f.h_recv = nullptr;
            if (!P.cudaaware) {
                CUDA_CHECK(cudaMallocHost(&f.h_send, f.bytes));
                CUDA_CHECK(cudaMallocHost(&f.h_recv, f.bytes));
            }
            faces.push_back(f);
        }
    }

    cudaStream_t sb, si;  // sb: boundary cells + communication, si: interior cells
    int prio_lo, prio_hi;  // numerically smaller value = higher priority
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&prio_lo, &prio_hi));
    CUDA_CHECK(cudaStreamCreateWithPriority(&sb, cudaStreamNonBlocking, P.prio ? prio_hi : prio_lo));
    CUDA_CHECK(cudaStreamCreateWithPriority(&si, cudaStreamNonBlocking, prio_lo));
    const int nblk = 256;
    double *d_part, h_part[256];
    CUDA_CHECK(cudaMalloc(&d_part, nblk * sizeof(double)));

    double t_comm = 0.0, t_check = 0.0, t_pack = 0.0, t_mpi = 0.0, t_unpack = 0.0;

    auto exchange = [&](double* arr) {
        CUDA_CHECK(cudaStreamSynchronize(sb));  // boundary cells are final
        double t = MPI_Wtime();
        for (auto& f : faces) {
            face_kernel<<<(f.count + 255) / 256, 256, 0, sb>>>(arr, g, f.dir, f.pack_plane,
                                                               f.d_send, 0, f.count);
            if (!P.cudaaware)
                CUDA_CHECK(cudaMemcpyAsync(f.h_send, f.d_send, f.bytes, cudaMemcpyDeviceToHost, sb));
        }
        CUDA_CHECK(cudaStreamSynchronize(sb));
        double t1 = MPI_Wtime();
        t_pack += t1 - t;
        std::vector<MPI_Request> req;
        for (auto& f : faces) {
            MPI_Request r;
            MPI_Irecv(P.cudaaware ? (void*)f.d_recv : (void*)f.h_recv, f.count, MPI_DOUBLE, f.nb,
                      f.rtag, cart, &r);
            req.push_back(r);
            MPI_Isend(P.cudaaware ? (void*)f.d_send : (void*)f.h_send, f.count, MPI_DOUBLE, f.nb,
                      f.stag, cart, &r);
            req.push_back(r);
        }
        MPI_Waitall((int)req.size(), req.data(), MPI_STATUSES_IGNORE);
        double t2 = MPI_Wtime();
        t_mpi += t2 - t1;
        for (auto& f : faces) {
            if (!P.cudaaware)
                CUDA_CHECK(cudaMemcpyAsync(f.d_recv, f.h_recv, f.bytes, cudaMemcpyHostToDevice, sb));
            face_kernel<<<(f.count + 255) / 256, 256, 0, sb>>>(arr, g, f.dir, f.unpack_plane,
                                                               f.d_recv, 1, f.count);
        }
        CUDA_CHECK(cudaStreamSynchronize(sb));
        double t3 = MPI_Wtime();
        t_unpack += t3 - t2;
        t_comm += t3 - t;
    };

    // ---- Jacobi loop
    double *u = d_u, *un = d_un, diff = -1.0;
    int it = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    CUDA_CHECK(cudaDeviceSynchronize());
    double t0 = MPI_Wtime();
    double t_setup = t0 - t_prog0;  // MPI/CUDA initialisation, allocation, source term
    while (it < P.iters) {
        bool chk = P.tol > 0.0 && ((it + 1) % P.check == 0);
        if (P.overlap) {
            jacobi_kernel<<<grid, blk, 0, sb>>>(u, un, d_f, g, 1);
            jacobi_kernel<<<grid, blk, 0, si>>>(u, un, d_f, g, 2);
            exchange(un);
            CUDA_CHECK(cudaStreamSynchronize(si));
        } else {
            jacobi_kernel<<<grid, blk, 0, sb>>>(u, un, d_f, g, 0);
            exchange(un);
        }
        ++it;
        if (chk) {
            double t = MPI_Wtime();
            maxdiff_kernel<<<nblk, 256, 0, sb>>>(un, u, g, d_part);
            CUDA_CHECK(cudaMemcpyAsync(h_part, d_part, nblk * sizeof(double),
                                       cudaMemcpyDeviceToHost, sb));
            CUDA_CHECK(cudaStreamSynchronize(sb));
            double m = 0.0;
            for (int b = 0; b < nblk; ++b) m = std::max(m, h_part[b]);
            MPI_Allreduce(&m, &diff, 1, MPI_DOUBLE, MPI_MAX, cart);
            t_check += MPI_Wtime() - t;
        }
        std::swap(u, un);
        if (chk && diff < P.tol) break;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double total = MPI_Wtime() - t0;
    double t_f0 = MPI_Wtime();

    // ---- error against the exact solution
    std::vector<double> hu(nelem);
    CUDA_CHECK(cudaMemcpy(hu.data(), u, nelem * sizeof(double), cudaMemcpyDeviceToHost));
    Norms loc;
    for (int k = g.hz; k < g.hz + g.lz; ++k)
        for (int j = 1; j <= g.ly; ++j)
            for (int i = 1; i <= g.lx; ++i) {
                double x = (g.offx + i) * g.h, y = (g.offy + j) * g.h;
                double z = (d == 3) ? (g.offz + k) * g.h : 0.0;
                size_t id = ((size_t)k * (g.ly + 2) + j) * (g.lx + 2) + i;
                loc.add(hu[id], exact_u(d, x, y, z));
            }
    Norms glob;
    MPI_Allreduce(&loc.sq_err, &glob.sq_err, 1, MPI_DOUBLE, MPI_SUM, cart);
    MPI_Allreduce(&loc.max_err, &glob.max_err, 1, MPI_DOUBLE, MPI_MAX, cart);
    MPI_Allreduce(&loc.sum_u, &glob.sum_u, 1, MPI_DOUBLE, MPI_SUM, cart);
    // slowest rank for every timer
    double tmax[9], tloc[9] = {total,   t_comm, t_pack, t_mpi, t_unpack, t_check, t_setup,
                               MPI_Wtime() - t_f0, MPI_Wtime() - t_prog0};
    MPI_Reduce(tloc, tmax, 9, MPI_DOUBLE, MPI_MAX, 0, cart);

    if (rank == 0) {
        Times tm;
        tm.total = tmax[0]; tm.comm = tmax[1]; tm.pack = tmax[2]; tm.mpi = tmax[3];
        tm.unpack = tmax[4]; tm.check = tmax[5]; tm.setup = tmax[6]; tm.final_ = tmax[7];
        tm.wall = tmax[8];
        report(P, "mpi_cuda", size, dims[0], dims[1], dims[2], it, tm, diff, glob, g.h);
    }

    for (auto& f : faces) {
        cudaFree(f.d_send);
        cudaFree(f.d_recv);
        if (!P.cudaaware) {
            cudaFreeHost(f.h_send);
            cudaFreeHost(f.h_recv);
        }
    }
    cudaFree(d_u); cudaFree(d_un); cudaFree(d_f); cudaFree(d_part);
    MPI_Finalize();
    return 0;
}
