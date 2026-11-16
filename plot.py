"""Make the report figures and summary tables from <dir>/*.csv (default: results).

usage: python3 analysis/plot.py [results_dir]      (needs numpy + pandas + matplotlib)

Time, problem size and number of cores / GPUs are drawn on logarithmic axes.
Speedup S = Ts / Tp, efficiency E = S / p, with Ts the time of the serial program on the same
problem and the same number of iterations (strong scaling); for weak scaling E = T(1) / T(p).
All times are the time of the Jacobi loop (total_s); setup and final error evaluation are
reported separately in the summary tables.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, NullFormatter

RES = sys.argv[1] if len(sys.argv) > 1 else "results"
PHASES = (("pack_s", "pack (+ GPU->host)"), ("mpi_s", "MPI"), ("unpack_s", "unpack (+ host->GPU)"))


def load(name):
    path = os.path.join(RES, name + ".csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def med(df, keys):
    return df.groupby(keys, as_index=False).median(numeric_only=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(RES, name), dpi=150, bbox_inches="tight")
    plt.close(fig)


def table(df, name):
    df.to_csv(os.path.join(RES, name), index=False)
    print(f"--- {name}")
    print(df.round(4).to_string(index=False))


def grid_label(r):
    g = f"{int(r.px)}x{int(r.py)}" + (f"x{int(r.pz)}" if r.dim == 3 else "")
    return g


def cut_label(r):
    return "".join(a for a, p in zip("xyz", (r.px, r.py, r.pz)) if p > 1) or "-"


def log2_axes(ax, x=True, y=False):
    """logarithmic axes with plain tick labels (1, 2, 4, ... instead of 2^0, 2^1, ...)"""
    if x:
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
    if y:
        ax.set_yscale("log", base=2)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())


# ======================================================== accuracy / iteration count (GPU, 1 rank)
a = load("accuracy")
if a is not None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for dim in (2, 3):
        g = a[a.dim == dim].sort_values("n")
        if g.empty:
            continue
        ax[0].loglog(g.n, g.l2_err, "o-", label=f"{dim}D")
        ax[1].loglog(g.n, g.iters, "o-", label=f"{dim}D")
    g2 = a[a.dim == 2].sort_values("n")
    if not g2.empty:
        n0, e0, i0 = g2.n.iloc[0], g2.l2_err.iloc[0], g2.iters.iloc[0]
        ax[0].loglog(g2.n, 0.6 * e0 * (n0 / g2.n) ** 2, "k--", label=r"$O(h^2)$")
        ax[1].loglog(g2.n, 0.6 * i0 * (g2.n / n0) ** 2, "k--", label=r"$O(n^2)$")
    ax[0].set_xlabel("n (points per dimension)"); ax[0].set_ylabel("L2 error")
    ax[1].set_xlabel("n (points per dimension)"); ax[1].set_ylabel("iterations to converge (tol 1e-12)")
    for x in ax:
        x.legend()
    save(fig, "accuracy.png")

# ======================================================== serial runtime vs problem size
s = load("serial")
if s is not None:
    s = s.copy()
    s["t_iter"] = s.total_s / s.iters
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    rows = []
    for a_, dim in zip(ax, (2, 3)):
        g = med(s[s.dim == dim], ["n"]).sort_values("n")
        if len(g) < 2:
            continue
        slope = np.polyfit(np.log(g.n), np.log(g.t_iter), 1)[0]
        a_.loglog(g.n, g.t_iter, "o-", label="measured")
        a_.loglog(g.n, g.t_iter.iloc[-1] * (g.n / g.n.iloc[-1]) ** dim, "k--", label=f"$O(n^{dim})$")
        a_.set_title(f"{dim}D serial, fitted slope {slope:.2f}")
        a_.set_xlabel("n (points per dimension)"); a_.set_ylabel("time per iteration (s)")
        a_.legend()
        for _, r in g.iterrows():
            rows.append(dict(dim=dim, n=int(r.n), points=int(r.n) ** dim, time_per_iter_s=r.t_iter,
                             loop_s=r.total_s, iters=int(r.iters), fitted_slope=slope))
    save(fig, "serial_scaling.png")
    table(pd.DataFrame(rows), "summary_serial.csv")


# ======================================================== MPI (CPU): strong scaling up to 256 cores
c = load("cpu_strong")
if c is not None:
    ser = c[c.version == "serial"]
    par = c[c.version == "mpi_cpu"]
    fig, ax = plt.subplots(3, 2, figsize=(10, 12))
    rows = []
    for col, dim in enumerate((2, 3)):
        for (n, it), grp in sorted(par[par.dim == dim].groupby(["n", "iters"])):
            g = med(grp, ["ranks"]).sort_values("ranks")
            ts = ser[(ser.dim == dim) & (ser.n == n) & (ser.iters == it)].total_s.median()
            if np.isnan(ts):
                continue
            sp = ts / g.total_s
            ef = sp / g.ranks
            line, = ax[0][col].plot(g.ranks, sp, "o-", label=f"n={n}")
            ax[1][col].plot(g.ranks, ef, "o-", label=f"n={n}")
            ax[2][col].plot(g.ranks, g.total_s, "o-", color=line.get_color(), label=f"n={n}")
            ax[2][col].plot(g.ranks, ts / g.ranks, "--", color=line.get_color(), alpha=0.5)
            for (_, r), s_, e_ in zip(g.iterrows(), sp, ef):
                rows.append(dict(dim=dim, n=int(n), iters=int(it), cores=int(r.ranks), loop_s=r.total_s,
                                 comm_s=r.comm_s, comm_fraction=r.comm_s / r.total_s,
                                 pack_s=r.pack_s, mpi_s=r.mpi_s, unpack_s=r.unpack_s,
                                 setup_s=r.setup_s, final_s=r.final_s, wall_s=r.wall_s,
                                 serial_s=ts, speedup=s_, efficiency=e_))
        pmax = par.ranks.max() if len(par) else 1
        ax[0][col].plot([1, pmax], [1, pmax], "k--", label="ideal")
        ax[1][col].axhline(1.0, color="k", ls="--", label="ideal")
        ax[0][col].set_title(f"{dim}D MPI (CPU), strong scaling: speedup")
        ax[1][col].set_title(f"{dim}D MPI (CPU), strong scaling: efficiency")
        ax[2][col].set_title(f"{dim}D MPI (CPU): loop time (dashed: ideal Ts/p)")
        for r_ in range(3):
            ax[r_][col].set_xlabel("MPI ranks = CPU cores")
            log2_axes(ax[r_][col], x=True, y=(r_ == 0))
            if r_ == 2:
                ax[r_][col].set_yscale("log")
            ax[r_][col].legend(fontsize=8)
        ax[0][col].set_ylabel("speedup  S = Ts / Tp")
        ax[1][col].set_ylabel("efficiency  E = S / p")
        ax[2][col].set_ylabel("loop time (s)")
    save(fig, "cpu_strong.png")
    table(pd.DataFrame(rows), "summary_cpu_strong.csv")

# ======================================================== MPI (CPU): weak scaling
w = load("cpu_weak")
if w is not None:
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    rows = []
    for col, dim in enumerate((2, 3)):
        for it, grp in sorted(w[w.dim == dim].groupby("iters")):
            g = med(grp, ["ranks", "n"]).sort_values("ranks")
            if g[g.ranks == 1].empty:
                continue
            t1 = g[g.ranks == 1].total_s.iloc[0]
            n0 = int(g[g.ranks == 1].n.iloc[0])
            eff = t1 / g.total_s
            ax[0][col].plot(g.ranks, g.total_s, "o-", label=f"{n0}^{dim} points per core")
            ax[1][col].plot(g.ranks, eff, "o-", label=f"{n0}^{dim} points per core")
            for (_, r), e_ in zip(g.iterrows(), eff):
                rows.append(dict(dim=dim, n=int(r.n), iters=int(it), cores=int(r.ranks),
                                 points_per_core=int(r.n) ** dim / r.ranks, loop_s=r.total_s,
                                 comm_s=r.comm_s, comm_fraction=r.comm_s / r.total_s,
                                 setup_s=r.setup_s, final_s=r.final_s, wall_s=r.wall_s, efficiency=e_))
        ax[0][col].set_title(f"{dim}D MPI (CPU), weak scaling: loop time")
        ax[1][col].set_title(f"{dim}D MPI (CPU), weak scaling: efficiency T(1)/T(p)")
        ax[1][col].axhline(1.0, color="k", ls="--")
        ax[0][col].set_yscale("log")
        for r_ in (0, 1):
            ax[r_][col].set_xlabel("MPI ranks = CPU cores")
            log2_axes(ax[r_][col])
            ax[r_][col].legend(fontsize=8)
        ax[0][col].set_ylabel("loop time (s)")
        ax[1][col].set_ylabel("efficiency  T(1) / T(p)")
    save(fig, "cpu_weak.png")
    table(pd.DataFrame(rows), "summary_cpu_weak.csv")

# ======================================================== MPI (CPU): strategies
t = load("cpu_strategies")
if t is not None:
    t = t.copy()
    t["compute_s"] = t.total_s - t.comm_s - t.check_s
    t["label"] = t.apply(lambda r: f"grid {grid_label(r)}, " + ("overlap" if r.overlap else "no overlap"),
                         axis=1)
    keys = sorted(t.groupby(["dim", "n", "ranks"]).groups)
    fig, axes = plt.subplots(1, len(keys), figsize=(4.6 * len(keys), 5.5), squeeze=False)
    for a_, (dim, n, p) in zip(axes[0], keys):
        g = med(t[(t.dim == dim) & (t.n == n) & (t.ranks == p)], ["label"]).sort_values("label")
        a_.bar(g.label, g.compute_s, label="compute")
        bottom = g.compute_s.copy()
        for col, name in PHASES:
            a_.bar(g.label, g[col], bottom=bottom, label=name.replace(" (+ GPU->host)", "")
                   .replace(" (+ host->GPU)", ""))
            bottom = bottom + g[col]
        a_.set_title(f"{dim}D n={n}, {p} cores"); a_.set_ylabel("loop time (s)")
        a_.tick_params(axis="x", rotation=75)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08))
    save(fig, "cpu_strategies.png")
    out = med(t, ["dim", "n", "ranks", "label"])[["dim", "n", "ranks", "label", "total_s", "compute_s",
                                                  "comm_s", "pack_s", "mpi_s", "unpack_s"]]
    table(out, "summary_cpu_strategies.csv")


def check_plot(df, name, title):
    g = med(df, ["check_every"]).sort_values("check_every")
    fig, ax = plt.subplots(figsize=(5, 4))
    x = [str(int(k)) for k in g.check_every]
    ax.bar(x, g.total_s - g.check_s, label="rest of the run")
    ax.bar(x, g.check_s, bottom=g.total_s - g.check_s, label="convergence checks")
    ax.set_xlabel("check every k iterations"); ax.set_ylabel("loop time (s)"); ax.set_title(title)
    ax.legend()
    save(fig, name)
    table(g[["check_every", "total_s", "check_s"]], "summary_" + name.replace(".png", ".csv"))


cc = load("cpu_check")
if cc is not None:
    check_plot(cc, "cpu_check.png", f"MPI (CPU), {int(cc.ranks.iloc[0])} cores")

# ======================================================== MPI+CUDA: strong scaling (1 -> 2 GPUs)
g_all = load("gpu_strong")
if g_all is not None:
    ser = g_all[g_all.version == "serial"]
    gpu = g_all[(g_all.version == "mpi_cuda") & (g_all.overlap == 0)]
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    rows = []
    for col, dim in enumerate((2, 3)):
        for n, grp in sorted(gpu[gpu.dim == dim].groupby("n")):
            one = grp[grp.ranks == 1].total_s.median()
            ts = ser[(ser.dim == dim) & (ser.n == n)].total_s.median()
            if np.isnan(one):
                continue
            for ca, g2 in grp[grp.ranks == 2].groupby("cudaaware"):
                two = g2.total_s.median()
                kind = "CUDA-aware" if ca else "host-staged"
                ls = "-" if not ca else ":"
                line, = ax[0][col].plot([1, 2], [1, one / two], "o" + ls, label=f"n={n}, {kind}")
                ax[1][col].plot([1, 2], [1, one / two / 2], "o" + ls, color=line.get_color(),
                                label=f"n={n}, {kind}")
                r = g2.median(numeric_only=True)
                rows.append(dict(dim=dim, n=int(n), cudaaware=int(ca), gpus=2, loop_s=two,
                                 loop_1gpu_s=one, comm_s=r.comm_s, pack_s=r.pack_s, mpi_s=r.mpi_s,
                                 unpack_s=r.unpack_s, setup_s=r.setup_s, final_s=r.final_s,
                                 wall_s=r.wall_s, speedup_vs_1gpu=one / two,
                                 efficiency_vs_1gpu=one / two / 2, serial_cpu_s=ts,
                                 speedup_vs_serial=ts / two, efficiency_vs_serial=ts / two / 2))
        ax[0][col].plot([1, 2], [1, 2], "k--", label="ideal")
        ax[1][col].axhline(1.0, color="k", ls="--", label="ideal")
        ax[0][col].set_title(f"{dim}D MPI+CUDA strong scaling: speedup")
        ax[1][col].set_title(f"{dim}D MPI+CUDA strong scaling: efficiency")
        for r_ in (0, 1):
            ax[r_][col].set_xlabel("GPUs (MPI ranks)")
            log2_axes(ax[r_][col], x=True, y=(r_ == 0))
            ax[r_][col].set_xticks([1, 2]); ax[r_][col].set_xticklabels(["1", "2"])
            ax[r_][col].minorticks_off()
            ax[r_][col].legend(fontsize=7, loc="upper left" if r_ == 0 else "best")
        ax[0][col].set_ylabel("speedup  T(1 GPU) / T(2 GPUs)")
        ax[1][col].set_ylabel("efficiency  S / 2")
        ax[0][col].set_yticks([1, 1.25, 1.5, 2]); ax[0][col].set_yticklabels(["1", "1.25", "1.5", "2"])
    save(fig, "gpu_strong.png")
    table(pd.DataFrame(rows), "summary_gpu_strong.csv")

    # runtime vs problem size: serial CPU, 1 GPU, 2 GPUs
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a_, dim in zip(ax, (2, 3)):
        sd = med(ser[ser.dim == dim], ["n"]).sort_values("n")
        if not sd.empty:
            a_.loglog(sd.n, sd.total_s, "o-", label="serial CPU")
        for p in (1, 2):
            for ca in (0, 1):
                gd = gpu[(gpu.dim == dim) & (gpu.ranks == p) & (gpu.cudaaware == ca)]
                if gd.empty or (p == 1 and ca == 1):
                    continue
                gd = med(gd, ["n"]).sort_values("n")
                a_.loglog(gd.n, gd.total_s, "o-",
                          label=f"{p} GPU" + ("s" if p == 2 else "") + (", CUDA-aware" if ca else ""))
        a_.set_title(f"{dim}D: loop time vs problem size")
        a_.set_xlabel("n (points per dimension)"); a_.set_ylabel("time (s)"); a_.legend(fontsize=8)
    save(fig, "gpu_time_vs_n.png")

# ======================================================== MPI+CUDA: strategies
t = load("gpu_strategies")
if t is not None:
    t = t.copy()
    t["compute_s"] = t.total_s - t.comm_s - t.check_s
    t["label"] = t.apply(
        lambda r: ("CUDA-aware" if r.cudaaware else "host-staged") + ", "
        + (f"overlap prio={int(r.prio)}" if r.overlap else "no overlap") + f", cut {cut_label(r)}", axis=1)
    keys = sorted(t.groupby(["dim", "n"]).groups)
    fig, axes = plt.subplots(1, len(keys), figsize=(5.4 * len(keys), 6.5), squeeze=False)
    for a_, (dim, n) in zip(axes[0], keys):
        g = med(t[(t.dim == dim) & (t.n == n)], ["label"]).sort_values("label")
        ov = g[g.label.str.contains("overlap prio")]
        no = g[~g.label.str.contains("overlap prio")]
        # without overlap the phases run one after the other and add up to the total
        a_.bar(no.label, no.compute_s, label="compute")
        bottom = no.compute_s.copy()
        for col, name in PHASES:
            a_.bar(no.label, no[col], bottom=bottom, label=name)
            bottom = bottom + no[col]
        # with overlap the phases run at the same time (and the exchange also waits for the
        # interior kernel), so only the total is meaningful
        a_.bar(ov.label, ov.total_s, color="0.65", hatch="//", label="overlap runs: total only")
        a_.set_title(f"{dim}D n={n}, 2 GPUs"); a_.set_ylabel("loop time (s)")
        a_.tick_params(axis="x", rotation=80, labelsize=8)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.22))
    save(fig, "gpu_strategies.png")
    out = med(t, ["dim", "n", "label"])[["dim", "n", "label", "total_s", "compute_s", "comm_s",
                                         "pack_s", "mpi_s", "unpack_s"]]
    table(out, "summary_gpu_strategies.csv")

gc = load("gpu_check")
if gc is not None:
    check_plot(gc, "gpu_check.png", "MPI+CUDA, 2 GPUs")

# ======================================================== MPI+CUDA: weak scaling (1 -> 2 GPUs)
w = load("gpu_weak")
if w is not None:
    rows = []
    for dim in (2, 3):
        d = w[w.dim == dim]
        n1s = sorted(d[d.ranks == 1].n.unique())
        n2s = sorted(d[d.ranks == 2].n.unique())
        for n1, n2 in zip(n1s, n2s):
            t1 = d[(d.ranks == 1) & (d.n == n1)].total_s.median()
            for ca, g2 in d[(d.ranks == 2) & (d.n == n2)].groupby("cudaaware"):
                t2 = g2.total_s.median()
                rows.append(dict(dim=dim, n_1gpu=int(n1), n_2gpu=int(n2), cudaaware=int(ca),
                                 t_1gpu=t1, t_2gpu=t2, comm_2gpu=g2.comm_s.median(), efficiency=t1 / t2))
    if rows:
        df = pd.DataFrame(rows)
        table(df, "summary_gpu_weak.csv")
        fig, ax = plt.subplots(figsize=(7, 4))
        cases = df[["dim", "n_1gpu", "n_2gpu"]].drop_duplicates().itertuples(index=False)
        cases = list(cases)
        width = 0.38
        for i, (dim, n1, n2) in enumerate(cases):
            for j, ca in enumerate((0, 1)):
                r = df[(df.dim == dim) & (df.n_1gpu == n1) & (df.cudaaware == ca)]
                if r.empty:
                    continue
                ax.bar(i + (j - 0.5) * width, r.efficiency.iloc[0], width, color=f"C{j}",
                       label=("CUDA-aware" if ca else "host-staged") if i == 0 else None)
        ax.set_xticks(range(len(cases)))
        ax.set_xticklabels([f"{d}D\n{a}->{b}" for d, a, b in cases])
        ax.axhline(1.0, color="k", ls="--")
        ax.set_ylabel("weak-scaling efficiency T(1 GPU)/T(2 GPUs)")
        ax.legend()
        save(fig, "gpu_weak.png")
print("figures and summaries written to", RES)
