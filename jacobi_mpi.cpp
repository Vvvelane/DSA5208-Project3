// MPI (CPU) Jacobi iteration for the 2D / 3D Poisson equation.
//
// Same design as the MPI+CUDA program: the global grid is split over a Cartesian process
// grid (px,py,pz), every rank holds its block plus a one-cell halo, and the halo layers are
// refreshed from the neighbours in every iteration. One MPI rank runs on one CPU core.
//   --px/--py/--pz  process-grid shape; --decomp 0 slab along the slowest axis,
//                   1 balanced (MPI_Dims_create, the default for this program)
//   --overlap       0: update everything, then exchange; 1: update the cells next to the
//                   faces first, start the exchange, update the interior, then wait
//   --check         how often the convergence test (MPI_Allreduce) runs
#include <mpi.h>
#include <algorithm>
#include <vector>
#include "common.h"

struct Face {
    int dir, side, nb, count, pack_plane, unpack_plane, stag, rtag;
    std::vector<double> send, recv;
};

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

    // ---- process grid
    int dims[3] = {P.px, P.py, P.pz};
    if (!dims[0] && !dims[1] && !dims[2]) {
        if (P.decomp == 0) {  // 1D slab along the slowest axis
            dims[0] = dims[1] = dims[2] = 1;
            dims[d - 1] = size;
        } else {              // balanced
            dims[0] = dims[1] = dims[2] = 0;
            if (d == 2) dims[2] = 1;
            MPI_Dims_create(size, 3, dims);
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

    // ---- local block
    const int hz = (d == 3) ? 1 : 0;
    const double h = 1.0 / (n + 1), h2 = h * h;
    int lx, ly, lz, offx, offy, offz = 0;
    split(n, dims[0], coords[0], lx, offx);
    split(n, dims[1], coords[1], ly, offy);
    if (d == 3) split(n, dims[2], coords[2], lz, offz);
    else lz = 1;
    if (lx < 1 || ly < 1 || lz < 1) {
        fprintf(stderr, "grid too small for this process grid\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const size_t sy = lx + 2, sz = sy * (ly + 2);
    const size_t nelem = sz * (lz + 2 * hz);
    const int kfirst = hz, klast = hz + lz - 1;
    std::vector<double> ua(nelem, 0.0), ub(nelem, 0.0), f(nelem, 0.0);
    for (int k = kfirst; k <= klast; ++k)
        for (int j = 1; j <= ly; ++j)
            for (int i = 1; i <= lx; ++i)
                f[k * sz + j * sy + i] = source_f(d, (offx + i) * h, (offy + j) * h,
                                                  d == 3 ? (offz + k) * h : 0.0);

    // ---- halo faces: direction dir in {x,y,z}, side 0 = low neighbour, 1 = high neighbour
    std::vector<Face> faces;
    for (int dir = 0; dir < d; ++dir) {
        int lo, hi;
        MPI_Cart_shift(cart, dir, 1, &lo, &hi);
        for (int side = 0; side < 2; ++side) {
            Face fc;
            fc.dir = dir;
            fc.side = side;
            fc.nb = side ? hi : lo;
            if (fc.nb == MPI_PROC_NULL) continue;
            int len = dir == 0 ? lx : dir == 1 ? ly : lz;
            fc.count = dir == 0 ? ly * lz : dir == 1 ? lx * lz : lx * ly;
            fc.pack_plane = side ? (dir == 2 ? klast : len) : (dir == 2 ? kfirst : 1);
            fc.unpack_plane = side ? (dir == 2 ? klast + 1 : len + 1) : (dir == 2 ? kfirst - 1 : 0);
            fc.stag = 2 * dir + side;        // tag of the face I send
            fc.rtag = 2 * dir + (1 - side);  // tag of the face my neighbour sends into this halo
            fc.send.assign(fc.count, 0.0);
            fc.recv.assign(fc.count, 0.0);
            faces.push_back(fc);
        }
    }

    // copy a face plane of arr to / from a contiguous buffer (same loop order on both sides)
    auto face_copy = [&](double* arr, const Face& fc, int plane, double* buf, bool unpack) {
        size_t t = 0;
        if (fc.dir == 0) {
            for (int k = kfirst; k <= klast; ++k)
                for (int j = 1; j <= ly; ++j) {
                    double& v = arr[k * sz + j * sy + plane];
                    if (unpack) v = buf[t++]; else buf[t++] = v;
                }
        } else if (fc.dir == 1) {
            for (int k = kfirst; k <= klast; ++k)
                for (int i = 1; i <= lx; ++i) {
                    double& v = arr[k * sz + plane * sy + i];
                    if (unpack) v = buf[t++]; else buf[t++] = v;
                }
        } else {
            for (int j = 1; j <= ly; ++j)
                for (int i = 1; i <= lx; ++i) {
                    double& v = arr[plane * sz + j * sy + i];
                    if (unpack) v = buf[t++]; else buf[t++] = v;
                }
        }
    };

    double t_comm = 0.0, t_check = 0.0, t_pack = 0.0, t_mpi = 0.0, t_unpack = 0.0;
    std::vector<MPI_Request> req;

    auto pack_all = [&](double* arr) {
        double t = MPI_Wtime();
        for (auto& fc : faces) face_copy(arr, fc, fc.pack_plane, fc.send.data(), false);
        double dt = MPI_Wtime() - t;
        t_pack += dt;
        t_comm += dt;
    };
    auto post_all = [&]() {
        double t = MPI_Wtime();
        req.clear();
        for (auto& fc : faces) {
            MPI_Request r;
            MPI_Irecv(fc.recv.data(), fc.count, MPI_DOUBLE, fc.nb, fc.rtag, cart, &r);
            req.push_back(r);
            MPI_Isend(fc.send.data(), fc.count, MPI_DOUBLE, fc.nb, fc.stag, cart, &r);
            req.push_back(r);
        }
        double dt = MPI_Wtime() - t;
        t_mpi += dt;
        t_comm += dt;
    };
    auto wait_unpack = [&](double* arr) {
        double t = MPI_Wtime();
        MPI_Waitall((int)req.size(), req.data(), MPI_STATUSES_IGNORE);
        double t2 = MPI_Wtime();
        for (auto& fc : faces) face_copy(arr, fc, fc.unpack_plane, fc.recv.data(), true);
        double t3 = MPI_Wtime();
        t_mpi += t2 - t;
        t_unpack += t3 - t2;
        t_comm += t3 - t;
    };

    // update the block i0..i1, j0..j1, k0..k1 (local interior indices); m = max |new - old|
    auto sweep = [&](const double* u, double* un, int i0, int i1, int j0, int j1, int k0, int k1,
                     bool chk, double& m) {
        const double c = 1.0 / (2.0 * d);
        for (int k = k0; k <= k1; ++k)
            for (int j = j0; j <= j1; ++j) {
                size_t base = k * sz + j * sy;
                if (chk) {
                    for (int i = i0; i <= i1; ++i) {
                        size_t id = base + i;
                        double s = u[id - 1] + u[id + 1] + u[id - sy] + u[id + sy];
                        if (d == 3) s += u[id - sz] + u[id + sz];
                        double v = (s - h2 * f[id]) * c;
                        un[id] = v;
                        m = std::max(m, std::fabs(v - u[id]));
                    }
                } else if (d == 3) {
                    for (int i = i0; i <= i1; ++i) {
                        size_t id = base + i;
                        un[id] = (u[id - 1] + u[id + 1] + u[id - sy] + u[id + sy] + u[id - sz] +
                                  u[id + sz] - h2 * f[id]) * c;
                    }
                } else {
                    for (int i = i0; i <= i1; ++i) {
                        size_t id = base + i;
                        un[id] = (u[id - 1] + u[id + 1] + u[id - sy] + u[id + sy] - h2 * f[id]) * c;
                    }
                }
            }
    };

    // ---- Jacobi loop
    double *u = ua.data(), *un = ub.data(), diff = -1.0;
    int it = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    double t_setup = t0 - t_prog0;
    const int z1 = (d == 3) ? 1 : 0;
    while (it < P.iters) {
        bool chk = P.tol > 0.0 && ((it + 1) % P.check == 0);
        double m = 0.0;
        if (P.overlap) {
            // cells next to a face first (they are sent), then the interior while the data moves
            sweep(u, un, 1, 1, 1, ly, kfirst, klast, chk, m);                       // x faces
            if (lx > 1) sweep(u, un, lx, lx, 1, ly, kfirst, klast, chk, m);
            sweep(u, un, 2, lx - 1, 1, 1, kfirst, klast, chk, m);                   // y faces
            if (ly > 1) sweep(u, un, 2, lx - 1, ly, ly, kfirst, klast, chk, m);
            if (d == 3) {                                                            // z faces
                sweep(u, un, 2, lx - 1, 2, ly - 1, kfirst, kfirst, chk, m);
                if (lz > 1) sweep(u, un, 2, lx - 1, 2, ly - 1, klast, klast, chk, m);
            }
            pack_all(un);
            post_all();
            sweep(u, un, 2, lx - 1, 2, ly - 1, kfirst + z1, klast - z1, chk, m);    // interior
            wait_unpack(un);
        } else {
            sweep(u, un, 1, lx, 1, ly, kfirst, klast, chk, m);
            pack_all(un);
            post_all();
            wait_unpack(un);
        }
        ++it;
        if (chk) {
            double t = MPI_Wtime();
            MPI_Allreduce(&m, &diff, 1, MPI_DOUBLE, MPI_MAX, cart);
            t_check += MPI_Wtime() - t;
        }
        std::swap(u, un);
        if (chk && diff < P.tol) break;
    }
    double total = MPI_Wtime() - t0;
    double t_f0 = MPI_Wtime();

    // ---- error against the exact solution
    Norms loc;
    for (int k = kfirst; k <= klast; ++k)
        for (int j = 1; j <= ly; ++j)
            for (int i = 1; i <= lx; ++i)
                loc.add(u[k * sz + j * sy + i],
                        exact_u(d, (offx + i) * h, (offy + j) * h, d == 3 ? (offz + k) * h : 0.0));
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
        report(P, "mpi_cpu", size, dims[0], dims[1], dims[2], it, tm, diff, glob, h);
    }
    MPI_Finalize();
    return 0;
}
