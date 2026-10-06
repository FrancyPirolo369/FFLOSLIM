#!/usr/bin/env python3
"""Integrale radiale in p della parte residua: Gauss-Legendre a pannelli spezzati agli spigoli.

Stessa funzione di test/p_integrand_probe.py, f(p) = p * int deps K [A_up A_dn - A0_up A0_dn], ma
integrata con pannelli GL i cui estremi sono i punti critici della riga:
    kF_up (con pannelli che si stringono attorno: kF_up +- W, +- W/4), Q + kF_dn, |Q - kF_dn|, 0, Lambda
e pannelli lunghi spezzati in tratti <= 0.5.  La funzione e' CALCOLATA nei nodi GL (A ricostruita da
Sigma in qualunque p), non interpolata.  Confronto con la griglia uniforme di produzione (31 nodi) e
con una uniforme fitta, per n nodi GL per pannello = 4, 6, 8, 16.

Uso (dalla radice di SLIM):  python3 test/p_gl_probe.py [RUN] [--w 0.15]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "test"))
import impi_union_probe as U  # noqa: E402
from p_integrand_probe import integrand  # noqa: E402

LAMBDA = 4.0
CASES = [(1.0, 0.3), (1.0, -0.05), (0.014, 0.3), (0.0, 0.3), (1.0, 1.0), (0.5, -0.3)]


def breakpoints(qv, kfu, kfd, w):
    b = {0.0, LAMBDA, kfu, kfu - w, kfu + w, kfu - w / 4, kfu + w / 4, qv + kfd, abs(qv - kfd)}
    b = np.array(sorted(x for x in b if 0.0 <= x <= LAMBDA))
    b = b[np.concatenate(([True], np.diff(b) > 1e-6))]
    out = [b[0]]
    for x in b[1:]:                       # pannelli lunghi spezzati in tratti <= 0.5
        n = int(np.ceil((x - out[-1]) / 0.5))
        out.extend(np.linspace(out[-1], x, n + 1)[1:])
    return np.array(out)


def gl_rule(bp, n):
    x, wt = np.polynomial.legendre.leggauss(n)
    nodes, weights = [], []
    for a, b in zip(bp[:-1], bp[1:]):
        nodes.append(0.5 * (b - a) * x + 0.5 * (b + a))
        weights.append(0.5 * (b - a) * wt)
    return np.concatenate(nodes), np.concatenate(weights)


def f_at(p, qv, om):
    U.G["p"] = np.asarray(p, float)
    return integrand(qv, om)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", default="P0p75_gmax")
    ap.add_argument("--w", type=float, default=0.15)
    ap.add_argument("--dw", type=float, default=2.5e-4)
    ap.add_argument("--np-fine", type=int, default=1601)
    a = ap.parse_args(argv)
    up = sorted(glob.glob(os.path.join(HERE, "out", "cluster", a.run, "iter*", "next_cubes", "A_komega_spinup_iter*.npz")))[-1]
    dn = up.replace("spinup", "spindown")
    work = os.path.join(HERE, "out", "impi_probe", "highP_integration", a.run)
    qff = U.setup(up, dn, os.path.join(work, "A0_up.npz"), os.path.join(work, "A0_dn.npz"),
                  n_p=31, dw=a.dw, n_linear=int(round(40 * 1e-3 / a.dw)))
    kfu, kfd = U.G["kf"]
    fig, axs = plt.subplots(2, 3, figsize=(16, 8.4), layout="constrained")
    print(f"{a.run}: kF_up = {kfu:.4f}, kF_dn = {kfd:.4f}, qff = {qff:.4f}; finestra attorno a kF_up +-{a.w}")
    hdr = f"{'fetta':22s} {'31 unif':>11s} {f'{a.np_fine} unif':>11s} " + " ".join(f"{f'GL{n}':>15s}" for n in (4, 6, 8, 16))
    print(hdr)
    for ax, (qr, om) in zip(axs.flat, CASES):
        qv = qr * qff
        p31 = np.linspace(0, LAMBDA, 31)
        i31 = np.trapezoid(f_at(p31, qv, om), p31)
        pf = np.linspace(0, LAMBDA, a.np_fine)
        ff = f_at(pf, qv, om)
        ifine = np.trapezoid(ff, pf)
        bp = breakpoints(qv, kfu, kfd, a.w)
        res = {}
        for n in (4, 6, 8, 16):
            x, wt = gl_rule(bp, n)
            fx = f_at(x, qv, om)
            res[n] = (float(np.sum(wt * fx)), x.size, x, fx)
        ref = res[16][0]
        cells = " ".join(f"{v[0]:+.5f}({v[1]:3d})" for n, v in res.items())
        print(f"Q={qr:5.3f}qff Om={om:+5.2f}  {i31:+.5f} {ifine:+.5f}  {cells}   "
              f"| err 31: {100 * (i31 / ref - 1):+6.1f}%, {a.np_fine}: {100 * (ifine / ref - 1):+6.2f}%, "
              f"GL4: {100 * (res[4][0] / ref - 1):+6.2f}%, GL8: {100 * (res[8][0] / ref - 1):+6.2f}%", flush=True)
        ax.plot(pf, ff, color="#1f1f1e", lw=1.0, label=f"{a.np_fine} uniformi: {ifine:+.4f}")
        ax.plot(p31, f_at(p31, qv, om), "o", color="#eb6834", ms=4, label=f"31 uniformi: {i31:+.4f}")
        ax.plot(res[6][2], res[6][3], "s", color="#2a78d6", ms=3, label=f"GL6 ({res[6][1]} nodi): {res[6][0]:+.4f}")
        for x in bp:
            ax.axvline(x, color="#2a78d6", lw=0.5, alpha=0.4)
        ax.set(xlim=(0, 2.4), xlabel="p = |k↑|", title=f"Q = {qr:g} qff, Ω = {om:+g}   (GL16 = {ref:+.4f})")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(alpha=0.3)
    dst = os.path.join(HERE, "out", "cluster", "plots", f"p_gl_{a.run}.png")
    fig.savefig(dst, dpi=110)
    print("scritto", dst)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
