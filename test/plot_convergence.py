#!/usr/bin/env python3
"""Convergenza per iterazione dai loop.log: una riga per P, colonne contact, g_c, gap del
minoritario (e Q*/qff ad alta P).  Ogni famiglia di run (etaexact, prod, gmax, x75, union...)
e' una curva, colore fisso per famiglia.  A bassa P le prod ripartono dall'ultima iterazione
delle etaexact: l'asse e' unico (etaexact 1-30, prod 31-...), con una linea al passaggio.  Ad alta
P ogni famiglia sta sul proprio indice (le catene hanno seed diversi).  I limiti in y ignorano
le prime due iterazioni di ogni run (transitorio del seed) e usano i percentili 2-98 (le
iterazioni rotte isolate, tipo Q* = 12 qff, escono dal riquadro); Q*/qff e' tagliato a [0, 1.15].
Ad alta P le etaexact non sono disegnate.

Due figure: <out>_lowP.png (P fino a 0.65) e <out>_highP.png (P da 0.70).
Uso (dalla radice di SLIM):
  python3 test/plot_convergence.py [--base out/cluster] [--out out/cluster/plots/convergence]
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
sys.path.insert(0, os.path.join(HERE, "test"))
from status_runs import parse  # noqa: E402

INK, MUTED, GRID, SURF = "#1f1f1e", "#6b6b68", "#e4e3df", "#fcfcfb"
# famiglia -> colore, ordine fisso (la stessa famiglia ha sempre lo stesso colore)
FAMILIES = [("etaexact", "#9a9a96"), ("prod", "#2a78d6"), ("gmax", "#eb6834"), ("x75a0p3", "#1baf7a"),
            ("x75a0p6", "#eda100"), ("prod_union", "#e87ba4"), ("prod_union_fine", "#4a3aa7")]
LOW = ["0p10", "0p20", "0p30", "0p40", "0p50", "0p60", "0p65"]
HIGH = ["0p70", "0p75", "0p80", "0p85", "0p90"]
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False, "font.size": 8.5,
    "legend.frameon": False, "lines.linewidth": 1.4,
})


def series(base, p, fam):
    log = os.path.join(base, f"P{p}_{fam}", "loop.log")
    if not os.path.exists(log):
        return None
    it, qff = parse(log)
    if not it:
        return None
    ks = sorted(it)
    q = np.array([it[k]["Q"] / qff if it[k].get("Q") is not None and qff else np.nan for k in ks])
    return dict(it=np.array(ks), C=np.array([it[k]["C"] for k in ks]),
                g=np.array([it[k]["g"] for k in ks]), gd=np.array([it[k]["gd"] for k in ks]), q=q)


def figure(base, pols, cols, path, offsets=None, skip=()):
    offsets = offsets or {}
    labels = {"C": "contact C", "g": "g_c", "gd": "gap_dn  [%]", "q": "Q* / qff"}
    fig, axs = plt.subplots(len(pols), len(cols), figsize=(3.6 * len(cols), 2.1 * len(pols)),
                            layout="constrained", squeeze=False)
    used = set()
    for r, p in enumerate(pols):
        lims = {key: [] for key in cols}
        for fam, col in FAMILIES:
            s = None if fam in skip else series(base, p, fam)
            if s is None:
                continue
            used.add(fam)
            x = s["it"] + offsets.get(fam, 0)
            for c, key in enumerate(cols):
                ax = axs[r, c]
                ax.plot(x, s[key], "-", color=col, lw=1.4, marker="o", ms=2.2, label=fam)
                lims[key].extend(v for v in s[key][2:] if np.isfinite(v))
                if key == "g" and fam in ("prod", "prod_union_fine"):
                    ax.annotate(f"{s['g'][-1]:+.3f}", (x[-1], s["g"][-1]), textcoords="offset points",
                                xytext=(4, 0), fontsize=7.5, color=INK, va="center")
        for c, key in enumerate(cols):
            if key == "q":
                axs[r, c].set_ylim(-0.05, 1.15)
            elif lims[key]:
                lo, hi = np.percentile(lims[key], [2, 98])
                pad = 0.08 * (hi - lo) if hi > lo else 0.05 * abs(hi) + 1e-3
                axs[r, c].set_ylim(lo - pad, hi + pad)
            if "prod" in offsets:
                axs[r, c].axvline(offsets["prod"] + 0.5, color=MUTED, lw=0.7, ls="--")
        axs[r, 0].set_ylabel(f"P = 0.{p[2:]}\n{labels[cols[0]]}")
        for c, key in enumerate(cols[1:], start=1):
            axs[r, c].set_ylabel(labels[key])
        if "gd" in cols:
            axs[r, cols.index("gd")].axhline(0, color=MUTED, lw=0.6)
    for c in range(len(cols)):
        axs[-1, c].set_xlabel("iterazione (etaexact 1-30, poi prod)" if offsets else "iterazione (di ciascuna run)")
    handles = [plt.Line2D([], [], color=col, marker="o", ms=3, lw=1.4, label=fam)
               for fam, col in FAMILIES if fam in used]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), fontsize=8.5)
    fig.savefig(path, dpi=110)
    print("scritto", path)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=os.path.join(HERE, "out", "cluster"))
    ap.add_argument("--out", default=os.path.join(HERE, "out", "cluster", "plots", "convergence"))
    a = ap.parse_args(argv)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    figure(a.base, LOW, ["C", "g", "gd"], f"{a.out}_lowP.png", offsets={"etaexact": 0, "prod": 30})
    # ad alta P le etaexact (pinning a qff, supercritiche: gap fino a +650%) schiacciano tutto
    figure(a.base, HIGH, ["C", "g", "gd", "q"], f"{a.out}_highP.png", skip=("etaexact",))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
