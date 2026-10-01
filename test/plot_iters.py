#!/usr/bin/env python3
"""Evoluzione per iterazione di n(k) e di Gamma^-1, una colonna per run.

Righe:
  n_up(k), n_dn(k)             la n portata all'iterazione dopo (mix alpha), it 0 = seed
  ReGamma^-1(Q,0) - max        contro Q/qff, in unita' di g (x 4 pi); pallino = Q selezionato
  ReGamma^-1(Q,0) grezzo       riferito al massimo dell'it 1 (stessa scala): cosa scende davvero
  ImGamma^-1(Q,Omega)          contro Omega a Q = 0 e a Q = qff (non dipende dallo shift)
Colore = iterazione (rampa chiaro -> scuro), uguale in tutte le righe.

Uso (dalla radice di SLIM):
  python3 test/plot_iters.py [RUN ...] [--base out/cluster] [--wlim 2] [--out ...]
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
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INK, MUTED, GRID, SURF = "#1f1f1e", "#6b6b68", "#e4e3df", "#fcfcfb"
RAMP = LinearSegmentedColormap.from_list(
    "iters", ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"])
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "legend.frameon": False, "lines.linewidth": 1.4,
})


def run_P(run):
    return float("0." + os.path.basename(run).split("_")[0][3:])


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", default=["P0p70_gmax", "P0p75_gmax"])
    ap.add_argument("--base", default=os.path.join(HERE, "out", "cluster"))
    ap.add_argument("--wlim", type=float, default=2.0, help="finestra in Omega per ImGamma^-1")
    ap.add_argument("--out", default=os.path.join(HERE, "out", "cluster", "plots", "iters_gmax.png"))
    a = ap.parse_args(argv)

    nrow, ncol = 6, len(a.runs)
    fig, ax = plt.subplots(nrow, ncol, figsize=(5.6 * ncol, 20), layout="constrained", squeeze=False)
    for c, run in enumerate(a.runs):
        d = os.path.join(a.base, run, "snap")
        its = sorted(int(f[-7:-4]) for f in glob.glob(os.path.join(d, "iter*.npz")))
        P = run_P(run)
        kf = {"up": np.sqrt(1 + P), "down": np.sqrt(1 - P)}
        col = {i: RAMP(i / max(its[-1], 1)) for i in [0] + its}
        ref = None
        s1 = np.load(os.path.join(d, f"sigma{its[0]:03d}.npz"))
        for r, spin, kmax in ((0, "up", 2.5), (1, "down", 3.0)):
            p = ax[r, c]
            k = s1["dr_k"]
            m = k <= kmax
            p.plot(k[m], s1[f"dr_seed_nk_{spin}"][m], color=col[0], label="it 0 (seed)")
            for i in its:
                s = np.load(os.path.join(d, f"sigma{i:03d}.npz"))
                # la n mescolata col mixing della run (alpha_0.3, alpha_0.6, ...; alpha_1 se e' 1)
                keys = [key for key in s.files if key.startswith("dr_alpha_") and key.endswith(f"_nk_{spin}")]
                mixed = [key for key in keys if key != f"dr_alpha_1_nk_{spin}"] or keys
                p.plot(k[m], s[mixed[0]][m], color=col[i], label=f"it {i}")
            p.axvline(kf[spin], color=MUTED, lw=0.8, ls=":")
            p.set(xlabel="k", ylabel=f"n_{'↑' if spin == 'up' else '↓'}(k)",
                  title=f"{run}: n_{'↑' if spin == 'up' else '↓'}(k)   (kF = {kf[spin]:.3f})")
        for i in its:
            z = np.load(os.path.join(d, f"iter{i:03d}.npz"))
            q, w, qff = np.asarray(z["q"]), np.asarray(z["omega"]), float(z["qff"])
            re, im = np.asarray(z["ReInvGamma"], float), np.asarray(z["ImInvGamma"], float)
            iw = int(np.argmin(np.abs(w)))
            re0 = re[:, iw]
            iq = int(np.nanargmax(re0))
            x = q / qff
            m = x <= 2.0
            ax[2, c].plot(x[m], 4 * np.pi * (re0[m] - re0[iq]), color=col[i], label=f"it {i}")
            ax[2, c].plot(x[iq], 0.0, "o", color=col[i], ms=7, mec=SURF, mew=1.0, zorder=5)
            ref = re0[iq] if ref is None else ref
            ax[3, c].plot(x[m], 4 * np.pi * (re0[m] - ref), color=col[i], label=f"it {i}")
            ax[3, c].plot(x[iq], 4 * np.pi * (re0[iq] - ref), "o", color=col[i], ms=7, mec=SURF, mew=1.0, zorder=5)
            mw = np.abs(w) <= a.wlim
            for r, qv, lab in ((4, 0.0, "Q = 0"), (5, qff, "Q = qff")):
                j = int(np.argmin(np.abs(q - qv)))
                ax[r, c].plot(w[mw], im[j, mw], color=col[i], label=f"it {i}")
                ax[r, c].set_title(f"{run}: ImΓ⁻¹({lab}, Ω)   (Q = {q[j]:.3f})")
        ax[2, c].axvline(1.0, color=MUTED, lw=0.8, ls=":")
        ax[2, c].set(xlabel="Q / qff", ylabel="4π [ReΓ⁻¹(Q,0) − max]  (= Δg)",
                     title=f"{run}: ReΓ⁻¹(Q, 0), pallino = Q selezionato")
        ax[2, c].set_ylim(-0.12, 0.005)
        ax[3, c].axvline(1.0, color=MUTED, lw=0.8, ls=":")
        ax[3, c].axhline(0, color=INK, lw=0.7)
        ax[3, c].set(xlabel="Q / qff", ylabel="4π [ReΓ⁻¹(Q,0) − max di it 1]",
                     title=f"{run}: ReΓ⁻¹(Q, 0) grezzo (rif. = max di it 1)")
        ax[3, c].set_ylim(-0.2, 0.01)
        for r in (4, 5):
            ax[r, c].axhline(0, color=INK, lw=0.7)
            ax[r, c].axvline(0, color=MUTED, lw=0.8, ls=":")
            ax[r, c].set(xlabel="Ω", ylabel="ImΓ⁻¹")
        for r in range(nrow):
            ax[r, c].legend(fontsize=7, loc="best", ncol=2)
    # stesse scale fra le colonne, per confrontare le due P a occhio
    for r in range(nrow):
        lo = min(ax[r, c].get_ylim()[0] for c in range(ncol))
        hi = max(ax[r, c].get_ylim()[1] for c in range(ncol))
        for c in range(ncol):
            ax[r, c].set_ylim(lo, hi)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print("scritto", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
