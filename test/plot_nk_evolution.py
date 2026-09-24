#!/usr/bin/env python3
"""Dove nasce l'eccesso di densita': evoluzione di n(k) per iterazione.

Uso:  python3 test/plot_nk_evolution.py RUN [BASE]     (BASE default out/cluster)
Legge BASE/RUN/snap/iterNNN.npz (test/extract_snapshot.py), scrive
BASE/plots/<RUN>_nk_evolution.png.  Righe: spin down, spin up.  Colonne:
  n(k) vicino al mare                    (gradino libero tratteggiato)
  k^2 [n(k) - th(kF-k)] / (mu/2)        in scala log k: l'area sotto la curva e'
                                         il contributo all'eccesso di densita'
  eccesso cumulativo  int_0^k k'[n - th] dk' / (mu/2):  a k -> inf e' il gap
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

INK, MUTED, GRID = "#1f1f1e", "#6b6b68", "#e4e3df"
RAMP = LinearSegmentedColormap.from_list(
    "iters", ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"])
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "legend.frameon": False, "lines.linewidth": 1.3,
})
MU = {"up": 1.65, "dn": 0.35}


def main(argv):
    run = argv[0]
    base = argv[1] if len(argv) > 1 else os.path.join("out", "cluster")
    F = sorted(glob.glob(os.path.join(base, run, "snap", "iter*.npz")))
    S = [np.load(f) for f in F]
    its = [int(os.path.basename(f)[4:7]) for f in F]
    fig, ax = plt.subplots(2, 3, figsize=(14, 7.8), layout="constrained")
    for row, spin in enumerate(("dn", "up")):
        mu = MU[spin]
        kf = np.sqrt(mu)
        half = mu / 2.0
        for j, (z, it) in enumerate(zip(S, its)):
            col = RAMP(j / max(len(S) - 1, 1))
            k = z["k_dn"] if spin == "dn" else z["k"]
            n = z["n_dn"] if spin == "dn" else z["n_up"]
            step = (k < kf).astype(float)
            m = k <= 2.5
            ax[row, 0].plot(k[m], n[m], color=col)
            mk = k > 0.05
            ax[row, 1].plot(k[mk], (k * k * (n - step))[mk] / half, color=col)
            cum = (np.concatenate(([0.0], np.cumsum(np.diff(k) * 0.5 * (k[1:] * n[1:] + k[:-1] * n[:-1]))))
                   - 0.5 * np.minimum(k, kf) ** 2) / half
            ax[row, 2].plot(k[mk], cum[mk], color=col, label=f"it {it}" if j in (0, len(S) - 1) else None)
        kk = np.linspace(0, 2.5, 500)
        ax[row, 0].plot(kk, (kk < kf).astype(float), color=MUTED, ls="--", lw=1)
        for a in ax[row]:
            a.axvline(kf, color=MUTED, ls=":", lw=1)
        for a in (ax[row, 1], ax[row, 2]):
            a.set_xscale("log")
            a.axhline(0, color=MUTED, lw=0.8)
            for kc in (3.0, 8.0):
                a.axvline(kc, color=MUTED, ls="-.", lw=0.7)
        lab = "down" if spin == "dn" else "up"
        ax[row, 0].set(title=f"n_{lab}(k)   (tratteggio: gradino libero, kF = {kf:.3f})",
                       xlabel="k", ylabel="n(k)")
        ax[row, 1].set(title=f"k^2 [n - th(kF-k)] / (mu/2)   spin {lab}  (area in ln k = eccesso)",
                       xlabel="k (log)", ylabel="per unita' di ln k")
        ax[row, 2].set(title=f"eccesso cumulativo fino a k, spin {lab}  (a destra = gap)",
                       xlabel="k (log)", ylabel="frazione di mu/2")
        ax[row, 2].legend(loc="upper left", fontsize=8)
    sm = plt.cm.ScalarMappable(cmap=RAMP, norm=plt.Normalize(its[0], its[-1]))
    fig.colorbar(sm, ax=ax, fraction=0.015, pad=0.01, label="iterazione")
    fig.suptitle(f"{run}: evoluzione di n(k)   (linee -. a k = 3 e 8)", color=INK)
    outd = os.path.join(base, "plots")
    os.makedirs(outd, exist_ok=True)
    dst = os.path.join(outd, f"{run}_nk_evolution.png")
    fig.savefig(dst, dpi=150)
    print("scritto", dst)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
