#!/usr/bin/env python3
"""Evoluzione di Sigma per iterazione (dati di test/extract_sigma.py).

Uso:  python3 test/plot_sigma_evolution.py RUN [BASE]    (BASE default out/cluster)
Scrive BASE/plots/<RUN>_sigma_evolution.png.  Righe: spin down, spin up.
  ImSigma(kF, w)          A(kF, w)          ReSigma(k, 0) - sigma0
  regola di somma int A dw - 1 per riga     densita' seed / fresca / mescolata
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
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]
RAMP = LinearSegmentedColormap.from_list(
    "iters", ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"])
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "legend.frameon": False, "lines.linewidth": 1.3,
})


def main(argv):
    run = argv[0]
    base = argv[1] if len(argv) > 1 else os.path.join("out", "cluster")
    F = sorted(glob.glob(os.path.join(base, run, "snap", "sigma*.npz")))
    if not F:
        print("nessun snap/sigmaNNN.npz: lancia prima test/extract_sigma.py")
        return 1
    S = [np.load(f) for f in F]
    its = np.array([int(os.path.basename(f)[5:8]) for f in F])
    fig, ax = plt.subplots(2, 5, figsize=(19, 7.5), layout="constrained")
    for row, (s, name) in enumerate((("dn", "down"), ("up", "up"))):
        mu = float(S[0][f"mu_{s}_{s}"]) if f"mu_{s}_{s}" in S[0].files else (0.35 if s == "dn" else 1.65)
        kf = np.sqrt(mu)
        for j, z in enumerate(S):
            col = RAMP(j / max(len(S) - 1, 1))
            rk = z[f"rows_k_{s}"]
            i = int(np.argmin(np.abs(rk - kf)))
            w = z[f"rows_w_{s}"][i]
            m = (w > -6) & (w < 6)
            ax[row, 0].plot(w[m], z[f"rows_ImS_{s}"][i][m], color=col)
            ax[row, 1].plot(w[m], z[f"rows_A_{s}"][i][m], color=col)
            k = z[f"k_{s}"]
            mk = k <= 8.0
            ax[row, 2].plot(k[mk], z[f"reS0_{s}"][mk] - float(z[f"sigma0_{s}"]), color=col)
            mk = k <= 12.0
            ax[row, 3].plot(k[mk], z[f"sumA_{s}"][mk] - 1.0, color=col)
        tgt = mu / 2.0
        for c, (key, lab) in enumerate((("seed", "seed (ingresso)"), ("alpha_1", "fresca (alpha = 1)"),
                                        ("alpha_0.3", "mescolata (emessa)"))):
            y = [float(z[f"dr_{key}_density_{name}"]) / tgt - 1 for z in S
                 if f"dr_{key}_density_{name}" in z.files]
            if y:
                ax[row, 4].plot(its[:len(y)], y, "o-", color=CAT[c], ms=4, label=lab)
        s0 = [float(z[f"sigma0_{s}"]) for z in S]
        ax[row, 0].set(title=f"ImSigma_{name}(k = {rk[i]:.3f} ~ kF, w)", xlabel="w")
        ax[row, 1].set(title=f"A_{name}(kF, w)", xlabel="w", yscale="log")
        ax[row, 2].set(title=f"ReSigma_{name}(k, 0) - sigma0   (sigma0: {s0[0]:+.3f} -> {s0[-1]:+.3f})",
                       xlabel="k")
        ax[row, 3].set(title=f"int A dw - 1 per riga, {name}", xlabel="k", ylim=(-0.03, 0.05))
        ax[row, 4].set(title=f"n_{name} / (mu/2) - 1", xlabel="iterazione")
        ax[row, 4].axhline(0, color=MUTED, lw=0.8)
        ax[row, 4].legend(fontsize=8)
        for a in ax[row, :4]:
            a.axhline(0, color=MUTED, lw=0.6)
        ax[row, 2].axvline(kf, color=MUTED, ls=":", lw=1)
        ax[row, 3].axvline(kf, color=MUTED, ls=":", lw=1)
    sm = plt.cm.ScalarMappable(cmap=RAMP, norm=plt.Normalize(its[0], its[-1]))
    fig.colorbar(sm, ax=ax[:, :4], fraction=0.012, pad=0.01, label="iterazione")
    fig.suptitle(f"{run}: Sigma per iterazione (cube emesse)", color=INK)
    outd = os.path.join(base, "plots")
    os.makedirs(outd, exist_ok=True)
    dst = os.path.join(outd, f"{run}_sigma_evolution.png")
    fig.savefig(dst, dpi=140)
    print("scritto", dst)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
