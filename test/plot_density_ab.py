#!/usr/bin/env python3
"""Confronto delle n(k) FRESCHE (alpha=1) di piu' passi di density.

Uso:  python3 test/plot_density_ab.py OUT.png ETICHETTA=density_results.npz [...]
      (la prima voce disegna anche il seed d'ingresso, in grigio)
Righe: spin down, spin up.  Colonne: n(k) vicino al mare; k^2 [n - th(kF-k)]/(mu/2)
in scala log k (l'area e' l'eccesso); eccesso cumulativo fino a k (a destra = gap).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID = "#1f1f1e", "#6b6b68", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "legend.frameon": False, "lines.linewidth": 1.4,
})
MU = {"down": 0.35, "up": 1.65}


def curves(ax, k, n, mu, color, label, ls="-"):
    kf, half = np.sqrt(mu), mu / 2.0
    step = (k < kf).astype(float)
    m = k <= 2.5
    ax[0].plot(k[m], n[m], color=color, ls=ls, label=label)
    mk = k > 0.05
    ax[1].plot(k[mk], (k * k * (n - step))[mk] / half, color=color, ls=ls)
    cum = (np.concatenate(([0.0], np.cumsum(np.diff(k) * 0.5 * (k[1:] * n[1:] + k[:-1] * n[:-1]))))
           - 0.5 * np.minimum(k, kf) ** 2) / half
    ax[2].plot(k[mk], cum[mk], color=color, ls=ls)
    return cum[-1]


def main(argv):
    dst, items = argv[0], [a.split("=", 1) for a in argv[1:]]
    fig, ax = plt.subplots(2, 3, figsize=(14, 7.8), layout="constrained")
    for row, spin in enumerate(("down", "up")):
        mu = MU[spin]
        for c, (lab, path) in enumerate(items):
            z = np.load(path)
            k = np.asarray(z["k"], float)
            if c == 0:
                g = curves(ax[row], k, np.asarray(z[f"seed_nk_{spin}"], float), mu, MUTED,
                           "seed (ingresso)", ls="--")
            g = curves(ax[row], k, np.asarray(z[f"alpha_1_nk_{spin}"], float), mu, CAT[c],
                       f"{lab}: gap {100 * (float(z[f'alpha_1_density_{spin}']) / (mu / 2) - 1):+.1f}%")
        kf = np.sqrt(mu)
        for a in ax[row]:
            a.axvline(kf, color=MUTED, ls=":", lw=1)
        for a in (ax[row, 1], ax[row, 2]):
            a.set_xscale("log")
            a.axhline(0, color=MUTED, lw=0.8)
            for kc in (3.0, 8.0):
                a.axvline(kc, color=MUTED, ls="-.", lw=0.7)
        ax[row, 0].set(title=f"n_{spin}(k) fresca (alpha = 1)", xlabel="k", ylabel="n(k)")
        ax[row, 0].legend(loc="upper right", fontsize=8)
        ax[row, 1].set(title=f"k^2 [n - th(kF-k)] / (mu/2), {spin}", xlabel="k (log)")
        ax[row, 2].set(title=f"eccesso cumulativo, {spin}  (a destra = gap)", xlabel="k (log)",
                       ylabel="frazione di mu/2")
    fig.suptitle("Un passo di density dallo stesso ingresso: n(k) fresche a confronto", color=INK)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    fig.savefig(dst, dpi=150)
    print("scritto", dst)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
