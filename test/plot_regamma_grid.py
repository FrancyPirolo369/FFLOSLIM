#!/usr/bin/env python3
"""ReGamma^-1(Q, Omega=0) - max, in unita' di delta, un pannello per run (ultima iterazione).

Punti = nodi Q della tabella; scala y indipendente per pannello.

Uso (dalla radice di SLIM):
  python3 test/plot_regamma_grid.py [RUN[:ITER] ...] [--base out/cluster] [--qmax 2] [--out ...]
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INK, MUTED, GRID, SURF, COL = "#1f1f1e", "#6b6b68", "#e4e3df", "#fcfcfb", "#2a78d6"
POLS = ["0p10", "0p20", "0p30", "0p40", "0p50", "0p60", "0p65", "0p70", "0p75", "0p85", "0p90"]
# per ogni P la prima famiglia che esiste: prima le run union (fine, poi normali), poi le altre
FAMILIES = ["prod_union_fine", "prod_union", "prod", "gmax", "x75a0p3"]


def default_runs(base):
    runs = []
    for p in POLS:
        for fam in FAMILIES:
            d = os.path.join(base, f"P{p}_{fam}")
            if glob.glob(os.path.join(d, "snap", "iter*.npz")):
                runs.append(f"P{p}_{fam}")
                break
    return runs
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
})


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", default=None,
                    help="default: per ogni P la prima fra " + ", ".join(FAMILIES))
    ap.add_argument("--base", default=os.path.join(HERE, "out", "cluster"))
    ap.add_argument("--qmax", type=float, default=2.0, help="Q/qff massimo")
    ap.add_argument("--ncol", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "cluster", "plots", "regamma_q_grid.png"))
    a = ap.parse_args(argv)
    if not a.runs:
        a.runs = default_runs(a.base)
    nrow = -(-len(a.runs) // a.ncol)
    fig, axs = plt.subplots(nrow, a.ncol, figsize=(4.0 * a.ncol, 3.0 * nrow), layout="constrained",
                            squeeze=False)
    for ax, spec in zip(axs.flat, a.runs):
        run, _, it = spec.partition(":")
        snaps = sorted(glob.glob(os.path.join(a.base, run, "snap", "iter*.npz")))
        if it:
            snaps = [s for s in snaps if s.endswith(f"iter{int(it):03d}.npz")]
        if not snaps:
            ax.set_visible(False)
            continue
        z = np.load(snaps[-1])
        q, w, qff = np.asarray(z["q"]), np.asarray(z["omega"]), float(z["qff"])
        re0 = np.asarray(z["ReInvGamma"], float)[:, int(np.argmin(np.abs(w)))]
        x = q / qff
        m = x <= a.qmax
        ax.plot(x[m], (re0[m] - np.nanmax(re0)) / 1e-3, "-o", color=COL, ms=2.5, lw=1.1)
        m = re.search(r"P0p(\d+)", run)
        ax.set_title(f"P = {float('0.' + m.group(1)):.2f}" if m else run, fontsize=10)
        print(f"{run}  it {int(snaps[-1][-7:-4])}")
        ax.set_xlim(0, a.qmax)
    for ax in axs[-1]:
        ax.set_xlabel("Q / qff")
    for ax in axs[:, 0]:
        ax.set_ylabel("ReΓ⁻¹(Q,0) − max  [δ]")
    for ax in list(axs.flat)[len(a.runs):]:
        ax.set_visible(False)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print("scritto", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
