#!/usr/bin/env python3
"""Atlante delle funzioni del minoritario ad alta P, dal cubo emesso (next_cubes) di una run.

Pannelli:
  1. A_dn(k, w) in scala log, con il polo di Dyson (dove w - xi - ReSigma + sigma0 = 0) e sigma0
  2. A_dn(k, w) su alcune righe k, con i NODI della griglia omega della riga (dove la riga e' campionata)
  3. ImSigma_dn(k, w) in scala log, con i nodi k dove Sigma e' davvero calcolata (snap sigma: ck_dn_k_sparse)
  4. ImSigma_dn su alcune righe k contro omega, con i nodi omega dove Sigma e' calcolata (ck_dn_omega)
  5. densita' occupata per k: k n(k) e la parte che viene da w < -3, -3..-1, -1..0
  6. regola di somma int A dw - 1 per k
Stampa: spaziatura dei nodi omega delle righe dove sta il ramo occupato profondo.

Uso (dalla radice di SLIM):  python3 test/highP_atlas.py RUN [IT]
"""
from __future__ import annotations

import glob
import math
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv):
    run = argv[0] if argv else "P0p90_x75a0p3"
    cubes = sorted(glob.glob(os.path.join(HERE, "out", "cluster", run, "iter*", "next_cubes", "A_komega_spindown_iter*.npz")))
    path = cubes[-1] if len(argv) < 2 else [c for c in cubes if f"iter{int(argv[1]):03d}" in c][0]
    it = int(os.path.basename(path).split("iter")[-1][:3])
    z = np.load(path, allow_pickle=True)
    k, w, A = z["k"], z["w"], z["A"]
    re, im = z["ReS"], z["ImS"]
    mu, s0, mass = float(z["mu_dn"]), float(z["sigma0"]), float(z["mass"])
    kf = math.sqrt(mass * mu)
    sg = os.path.join(HERE, "out", "cluster", run, "snap", f"sigma{it:03d}.npz")
    s = np.load(sg, allow_pickle=True) if os.path.exists(sg) else None
    kmax_plot = 4.0 * kf if kf > 0.4 else 8 * kf
    kmax_plot = min(max(kmax_plot, 2.5), 4.0)
    wmin, wmax = -14.0, 4.0
    fig, axs = plt.subplots(2, 3, figsize=(17, 9.5), layout="constrained")
    # 1. mappa A
    ax = axs[0, 0]
    sel = k <= kmax_plot
    for i in np.flatnonzero(sel)[::2]:
        m = (w[i] > wmin) & (w[i] < wmax)
        ax.scatter(np.full(m.sum(), k[i]), w[i][m], c=np.maximum(A[i][m], 1e-4), s=1.5, cmap="viridis",
                   norm=LogNorm(1e-4, 10), rasterized=True)
    # polo di Dyson
    poles_k, poles_w = [], []
    for i in np.flatnonzero(sel):
        f = w[i] - (k[i] ** 2 / mass - mu) - (re[i] - s0)
        for j in np.flatnonzero(np.sign(f[:-1]) * np.sign(f[1:]) < 0):
            if wmin < w[i][j] < wmax:
                poles_k.append(k[i]); poles_w.append(w[i][j])
    ax.plot(poles_k, poles_w, ".", color="#eb6834", ms=1.5, label="zeri di ω − ξ − ReΣ + σ0")
    ax.axhline(0, color="w", lw=0.6)
    ax.axvline(kf, color="w", lw=0.6, ls="--")
    ax.set(xlabel="k", ylabel="ω", ylim=(wmin, wmax), title=f"{run} it{it}: A↓(k, ω) (log), σ0 = {s0:+.2f}")
    ax.legend(fontsize=7, loc="lower right")
    # 2. righe di A con i nodi
    ax = axs[0, 1]
    cols = ["#1f1f1e", "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
    for c, kr in zip(cols, (0.5, 1.0, 1.5, 2.5, 4.0)):
        i = int(np.argmin(np.abs(k - kr * kf)))
        m = (w[i] > wmin) & (w[i] < wmax)
        ax.semilogy(w[i][m], np.maximum(A[i][m], 1e-6), "-", color=c, lw=1.0, label=f"k = {k[i] / kf:.2f} kF")
        ax.semilogy(w[i][m], np.maximum(A[i][m], 1e-6), "o", color=c, ms=2)
        neg = m & (w[i] < -1.0)
        if neg.any():
            dws = np.diff(w[i][neg])
            print(f"  riga k = {k[i] / kf:.2f} kF: picco occupato piu' alto a w = {w[i][neg][np.argmax(A[i][neg])]:+.2f}, "
                  f"passo omega nella zona w < -1: da {dws.min():.3f} a {dws.max():.3f} (mediana {np.median(dws):.3f})")
    ax.set(xlabel="ω", ylabel="A↓(k, ω)", ylim=(1e-5, 50), title="righe di A↓ con i nodi della griglia ω della riga")
    ax.legend(fontsize=7)
    # 3. mappa ImSigma
    ax = axs[0, 2]
    for i in np.flatnonzero(sel)[::2]:
        m = (w[i] > wmin) & (w[i] < wmax)
        ax.scatter(np.full(m.sum(), k[i]), w[i][m], c=np.maximum(-im[i][m], 1e-3), s=1.5, cmap="magma",
                   norm=LogNorm(1e-3, 30), rasterized=True)
    if s is not None:
        for kn in np.asarray(s["ck_dn_k_sparse"]).ravel():
            if kn <= kmax_plot:
                ax.axvline(kn, color="c", lw=0.6)
    ax.set(xlabel="k", ylabel="ω", ylim=(wmin, wmax), title="−ImΣ↓(k, ω) (log); azzurro: k dove Σ è calcolata")
    # 4. righe di ImSigma con i nodi omega di Sigma
    ax = axs[1, 0]
    for c, kr in zip(cols, (0.5, 1.0, 1.5, 2.5, 4.0)):
        i = int(np.argmin(np.abs(k - kr * kf)))
        m = (w[i] > wmin) & (w[i] < wmax)
        ax.plot(w[i][m], -im[i][m], "-", color=c, lw=1.1, label=f"k = {k[i] / kf:.2f} kF")
    if s is not None:
        for wn in np.asarray(s["ck_dn_omega"]).ravel():
            if wmin < wn < wmax:
                ax.axvline(wn, color="#9a9a96", lw=0.4)
    ax.set(xlabel="ω", ylabel="−ImΣ↓", xlim=(wmin, wmax), title="ImΣ↓ su alcune righe (grigio: nodi ω dove Σ è calcolata)")
    ax.legend(fontsize=7)
    # 5. densita' occupata per k e per fascia di omega
    ax = axs[1, 1]
    bands = [(-1e9, -3, "ω < −3"), (-3, -1, "−3 < ω < −1"), (-1, 0.0001, "−1 < ω < 0")]
    tot = np.array([np.trapezoid(np.where(w[i] <= 0, A[i], 0), w[i]) for i in range(k.size)])
    ax.plot(k / kf, k * tot, color="#1f1f1e", lw=1.6, label="k n(k) totale")
    for c, (lo, hi, lab) in zip(cols[1:], bands):
        nb = np.array([np.trapezoid(np.where((w[i] > lo) & (w[i] <= hi) & (w[i] <= 0), A[i], 0), w[i]) for i in range(k.size)])
        ax.plot(k / kf, k * nb, color=c, lw=1.2, label=lab)
    ax.set(xlabel="k / kF↓", ylabel="k n(k)", xlim=(0, kmax_plot / kf), title="da dove viene n↓: per k e per fascia di ω")
    ax.legend(fontsize=7)
    # 6. regola di somma
    ax = axs[1, 2]
    sr = np.array([np.trapezoid(A[i], w[i]) for i in range(k.size)])
    ax.plot(k / kf, sr - 1, color="#2a78d6", lw=1.2)
    ax.set(xlabel="k / kF↓", ylabel="∫A↓ dω − 1", xlim=(0, kmax_plot / kf), title="regola di somma del minoritario")
    for ax in axs.flat:
        ax.grid(alpha=0.25)
    dst = os.path.join(HERE, "out", "cluster", "plots", f"highP_atlas_{run}.png")
    fig.savefig(dst, dpi=105)
    print("scritto", dst)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
