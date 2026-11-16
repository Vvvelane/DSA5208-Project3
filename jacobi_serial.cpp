// Serial Jacobi iteration for the 2D / 3D Poisson equation (CPU baseline).
#include <algorithm>
#include <chrono>
#include <vector>
#include "common.h"

int main(int argc, char** argv) {
    Params P = parse_args(argc, argv);
    auto tp0 = std::chrono::steady_clock::now();
    auto since = [](std::chrono::steady_clock::time_point a) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
    };
    const int d = P.dim, n = P.n;
    const int hz = (d == 3) ? 1 : 0;
    const int lz = (d == 3) ? n : 1;
    const size_t nx = n + 2, ny = n + 2, nz = lz + 2 * hz;
    const double h = 1.0 / (n + 1), h2 = h * h;

    std::vector<double> u(nx * ny * nz, 0.0), un(nx * ny * nz, 0.0), f(nx * ny * nz, 0.0);
    for (int k = hz; k < hz + lz; ++k)
        for (int j = 1; j <= n; ++j)
            for (int i = 1; i <= n; ++i)
                f[(k * ny + j) * nx + i] =
                    source_f(d, i * h, j * h, d == 3 ? k * h : 0.0);

    Times tm;
    tm.setup = since(tp0);
    double diff = -1.0;
    int it = 0;
    auto t0 = std::chrono::steady_clock::now();
    while (it < P.iters) {
        bool chk = P.tol > 0.0 && ((it + 1) % P.check == 0);
        double m = 0.0;
        for (int k = hz; k < hz + lz; ++k)
            for (int j = 1; j <= n; ++j)
                for (int i = 1; i <= n; ++i) {
                    size_t id = (k * ny + j) * nx + i;
                    double s = u[id - 1] + u[id + 1] + u[id - nx] + u[id + nx];
                    if (d == 3) s += u[id - nx * ny] + u[id + nx * ny];
                    double v = (s - h2 * f[id]) / (2.0 * d);
                    un[id] = v;
                    if (chk) m = std::max(m, std::fabs(v - u[id]));
                }
        std::swap(u, un);
        ++it;
        if (chk) {
            diff = m;
            if (m < P.tol) break;
        }
    }
    tm.total = since(t0);

    auto tf0 = std::chrono::steady_clock::now();
    Norms nm;
    for (int k = hz; k < hz + lz; ++k)
        for (int j = 1; j <= n; ++j)
            for (int i = 1; i <= n; ++i)
                nm.add(u[(k * ny + j) * nx + i], exact_u(d, i * h, j * h, d == 3 ? k * h : 0.0));

    tm.final_ = since(tf0);
    tm.wall = since(tp0);
    report(P, "serial", 1, 1, 1, 1, it, tm, diff, nm, h);
    return 0;
}
