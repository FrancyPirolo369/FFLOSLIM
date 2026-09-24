#!/usr/bin/env python3
"""Grafico della sonda test/sigma_resolution_probe.py (+ il mixing di D).

Uso:  python3 test/plot_probe.py PROBE_NPZ SEED_DOWN [SNAP_DIR_D] [OUT_PNG]
  (a,b) ImSigma(k, w) esatta contro la ricostruzione Pchip dai nodi omega di
        produzione (41) e dal default del motore (97)
  (c)   n(k) k^4 di Tan riga per riga per sigma_nomega = fitta, 41, 97, 240
  (d)   n k^4 / C a k=6 per iterazione di D contro 1 - (1-alpha)^n
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#1f1f1e", "#6b6b68", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "legend.frameon": False, "lines.linewidth": 1.4,
})


def eval_nodes(w_base, n):
    wu = w_base[(w_base >= -216.0 - 1e-12) & (w_base <= 144.0 + 1e-12)]
    idx = np.unique(np.round(np.linspace(0, wu.size - 1, min(n, wu.size))).astype(int))
    return wu[idx]


def main(argv):
    probe, seed_dn = argv[0], argv[1]
    snap = argv[2] if len(argv) > 2 else None
    dst = argv[3] if len(argv) > 3 else os.path.join("out", "cluster", "plots",
                                                        "sigma_nomega_probe.png")
    z = np.load(probe)
    k, w, im = z["k"], z["omega"], z["im"]
    mu, s0 = float(z["mu_ext"]), float(z["sigma0"])
    w_base = np.asarray(np.load(seed_dn)["w"], dtype=float)
    truth = [PchipInterpolator(w, im[i]) for i in range(k.size)]

    def nodes_for(n):
        we = eval_nodes(w_base, n)
        return we[(we >= w[0]) & (we <= w[-1])]

    def tan(i, nodes):
        xi = k[i] ** 2 - (mu + s0)
        wf = np.linspace(max(nodes[0], -(k[i] ** 2) - 60.0), 0.0, 40001)
        v = truth[i](nodes)
        f = np.nan_to_num(np.minimum(PchipInterpolator(nodes, v, extrapolate=False)(wf), 0.0))
        return k[i] ** 4 * np.trapezoid(-f / np.pi / (wf - xi) ** 2, wf)

    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5), layout="constrained")
    wf = np.linspace(-95.0, 0.0, 4000)
    for a, kk in zip((ax[0, 0], ax[0, 1]), (5.5, 7.0)):
        i = int(np.argmin(np.abs(k - kk)))
        a.plot(wf, truth[i](wf), color=CAT[0], lw=2.0, label="esatta (omega fitta)")
        for c, n, mk in ((CAT[1], 41, "o"), (CAT[2], 97, "s")):
            nd = nodes_for(n)
            nd = nd[(nd >= wf[0]) & (nd <= 0.0)]
            rec = PchipInterpolator(nd, truth[i](nd), extrapolate=False)(wf)
            a.plot(wf, rec, color=c, lw=1.2, ls="--",
                   label=f"sigma_nomega = {n}  (errore n k^4: {tan(i, nodes_for(n)) / tan(i, w) - 1:+.1%})")
            a.plot(nd, truth[i](nd), mk, color=c, ms=5 if n == 41 else 3.5)
        a.axvline(mu - k[i] ** 2 + 1.3, color=MUTED, ls=":", lw=1)   # ~ mu_int - k^2
        a.set(title=f"ImSigma_down(k = {k[i]:g}, w)   (punti = dove Sigma e' calcolata)",
              xlabel="w  (lato occupato: e' questo che fa n(k))", ylabel="ImSigma", xlim=(-95, 3))
        a.legend(loc="lower left", fontsize=8)

    a = ax[1, 0]
    for c, n, lab, mk in ((CAT[0], None, "esatta (omega fitta)", "o"), (CAT[1], 41, "41 (test oggi)", "o"),
                          (CAT[2], 97, "97 (default motore, G)", "s"), (CAT[3], 240, "240", "^")):
        y = [tan(i, w if n is None else nodes_for(n)) for i in range(k.size)]
        a.plot(k, y, color=c, marker=mk, ms=5, lw=2.0 if n is None else 1.2, label=lab)
    a.set(title="n(k) k^4 di Tan, riga per riga (seed + pair buona, PINTMAX 12)",
          xlabel="k", ylabel="n(k) k^4")
    a.legend(loc="upper left", fontsize=8)

    a = ax[1, 1]
    if snap:
        F = sorted(glob.glob(os.path.join(snap, "iter*.npz")))
        it, r = [], []
        for f in F:
            s = np.load(f)
            kk = s["k_dn"]
            it.append(int(os.path.basename(f)[4:7]))
            r.append(np.interp(6.0, kk, s["n_dn"] * kk ** 4) / float(s["pair_contact"]))
        it = np.array(it)
        a.plot(it, r, "o-", color=CAT[0], ms=5, label="D: n k^4 / C a k = 6")
        a.plot(it, 1 - 0.7 ** it, "s--", color=CAT[1], ms=4, label="1 - 0.7^n  (mixing alpha = 0.3)")
        a.set(title="la coda di D si riempie al ritmo del mixing", xlabel="iterazione",
              ylabel="n k^4 / C", ylim=(0, 1.05))
        a.legend(loc="lower right", fontsize=8)
    else:
        a.set_visible(False)
    fig.suptitle("Wiggles di n(k)k^4: il satellite di Tan campionato su 2-3 nodi omega", color=INK)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    fig.savefig(dst, dpi=150)
    print("scritto", dst)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
