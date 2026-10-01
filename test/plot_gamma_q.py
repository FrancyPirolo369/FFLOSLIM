#!/usr/bin/env python3
"""ReGamma^-1(Q, Omega) dalle istantanee snap/iterNNN.npz, per vedere dove sta il massimo in Q.

Tutto in unita' di g (x 4 pi): una differenza di ReGamma^-1 vale quella stessa differenza in g_c.
  (a) ReGamma^-1(Q, 0) - max_Q, contro Q/qff: il massimo e' il punto di Thouless del global-max
  (b) ReGamma^-1(0, Omega) - shift        canale pBCS lungo Omega
  (c) ReGamma^-1(Q*, Omega) - shift       Q* = massimo di (a) per ogni run
  (d) mappa di ReGamma^-1(Q, Omega) - shift per l'ultima run, contorno nero = zero
shift = massimo di ReGamma^-1(Q, 0) sulla griglia (il pinning global-max), anche per le run
pinnate a qff, cosi' tutte sono confrontate allo stesso criterio.

Uso (dalla radice di SLIM):
  python3 test/plot_gamma_q.py [RUN[:ITER] ...] [--base out/cluster] [--out ...]
RUN e' una cartella sotto --base; senza :ITER prende l'ultima istantanea.
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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELTA = 1e-3
INK, MUTED, GRID, SURF = "#1f1f1e", "#6b6b68", "#e4e3df", "#fcfcfb"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
DIV = LinearSegmentedColormap.from_list("div", ["#184f95", "#2a78d6", "#f0efec", "#e34948", "#8f1f1f"])
DEFAULT = ["P0p65_etaexact", "P0p65_prod", "P0p70_etaexact", "P0p70_gmax", "P0p75_gmax"]
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "legend.frameon": False, "lines.linewidth": 1.6,
})


def load(base, spec):
    run, _, it = spec.partition(":")
    snaps = sorted(glob.glob(os.path.join(base, run, "snap", "iter*.npz")))
    if it:
        snaps = [s for s in snaps if s.endswith(f"iter{int(it):03d}.npz")]
    if not snaps:
        raise SystemExit(f"nessuna istantanea per {spec}")
    z = np.load(snaps[-1])
    q, w = np.asarray(z["q"], float), np.asarray(z["omega"], float)
    re = np.asarray(z["ReInvGamma"], float)
    iw = int(np.argmin(np.abs(w)))
    iq = int(np.nanargmax(re[:, iw]))
    return dict(run=run, it=int(snaps[-1][-7:-4]), q=q, w=w, re=re, im=np.asarray(z["ImInvGamma"], float),
                qff=float(z["qff"]), iw=iw, iq=iq, shift=re[iq, iw])


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", default=DEFAULT)
    ap.add_argument("--base", default=os.path.join(HERE, "out", "cluster"))
    ap.add_argument("--wlim", type=float, default=1.0, help="finestra in Omega per (b), (c), (d)")
    ap.add_argument("--qmax", type=float, default=2.0, help="Q/qff massimo in (a) e (d)")
    ap.add_argument("--out", default=os.path.join(HERE, "out", "cluster", "plots", "gamma_q.png"))
    a = ap.parse_args(argv)
    R = [load(a.base, s) for s in a.runs]

    fig, ax = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    (pa, pm), (pb, pc) = ax
    print(f"{'run':18s} it   qff    Q*/qff   Q=0 - max (g)   qff - max (g)")
    for r, col in zip(R, CAT):
        x = r["q"] / r["qff"]
        prof = 4 * np.pi * (r["re"][:, r["iw"]] - r["shift"])
        lab = f"{r['run']} it {r['it']}"
        m = x <= a.qmax
        pa.plot(x[m], prof[m], "-", color=col, label=lab)
        pa.plot(x[r["iq"]], 0.0, "o", color=col, ms=8, mec=SURF, mew=1.2, zorder=5)
        iqff = int(np.argmin(np.abs(x - 1.0)))
        print(f"{r['run']:18s} {r['it']:3d}  {r['qff']:.4f}  {x[r['iq']]:6.3f}   {prof[0]:+.4f}          {prof[iqff]:+.4f}")
        mw = np.abs(r["w"]) <= a.wlim
        for p, i in ((pb, 0), (pc, r["iq"])):
            p.plot(r["w"][mw], 4 * np.pi * (r["re"][i, mw] - r["shift"]), "-", color=col, label=lab)
    pa.axhspan(-4 * np.pi * DELTA, 0, color=GRID, zorder=0)
    pa.axvline(1.0, color=MUTED, lw=0.8, ls=":")
    pa.set(xlabel="Q / qff", ylabel="4π [ReΓ⁻¹(Q,0) − max]   (= Δg)",
           title="(a) Ω = 0: il massimo (pallino) e' il Thouless global-max; banda = δ")
    pa.set_ylim(bottom=max(pa.get_ylim()[0], -0.25))
    pa.legend(fontsize=8, loc="lower right")
    for p, t in ((pb, "(b) Q = 0 (pBCS)"), (pc, "(c) Q = Q* (massimo di (a))")):
        p.axhline(0, color=INK, lw=0.8)
        p.axvline(0, color=MUTED, lw=0.8, ls=":")
        p.set(xlabel="Ω", ylabel="4π [ReΓ⁻¹ − shift]", title=t)
        p.set_ylim(-0.25, 0.15)

    r = R[-1]
    mq, mw = r["q"] / r["qff"] <= a.qmax, np.abs(r["w"]) <= a.wlim
    Z = 4 * np.pi * (r["re"][np.ix_(mq, mw)] - r["shift"])
    X, Y = np.meshgrid(r["w"][mw], r["q"][mq] / r["qff"])
    lim = 0.15
    cs = pm.pcolormesh(X, Y, np.clip(Z, -lim, lim), cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim), shading="nearest")
    pm.contour(X, Y, Z, levels=[0.0], colors=INK, linewidths=1.0)
    pm.axvline(0, color=MUTED, lw=0.8, ls=":")
    pm.plot(0, r["q"][r["iq"]] / r["qff"], "o", color=INK, ms=6)
    pm.set(xlabel="Ω", ylabel="Q / qff", title=f"(d) {r['run']} it {r['it']}: 4π [ReΓ⁻¹(Q,Ω) − shift]")
    pm.grid(False)
    fig.colorbar(cs, ax=pm, shrink=0.9, label="rosso > 0 (sopra soglia), blu < 0")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print("scritto", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
