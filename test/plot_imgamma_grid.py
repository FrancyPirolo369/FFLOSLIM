#!/usr/bin/env python3
"""Gamma^-1(Q = qff, Omega) vicino a Omega = 0, un pannello per run (ultima iterazione).

--part im (default), due figure:
  <out>.png      lineare, ImGamma^-1 / delta contro Omega, punti = nodi della tabella
  <out>_log.png  log-log, |ImGamma^-1| contro |Omega|, i due lati separati, con rette guida
                 di pendenza 1/3, 1/2, 2/3
--part re, due figure:
  <out>.png      lineare, [ReGamma^-1(Omega) - ReGamma^-1(0)] / delta contro Omega
  <out>_log.png  la stessa contro |Omega| in scala log, i due lati separati; retta guida
                 -(D/pi) ln|Omega| = quello che la KK da' a un gradino D = ImGamma^-1(+ws) - ImGamma^-1(-ws)
Stessa scelta delle run di plot_regamma_grid.py (union fine, union, prod, gmax, x75a0p3).

Uso (dalla radice di SLIM):
  python3 test/plot_imgamma_grid.py [RUN[:ITER] ...] [--part im|re] [--wlim 0.15] [--out ...]
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_regamma_grid import FAMILIES, HERE, default_runs  # noqa: E402  (imposta anche lo stile)

DELTA = 1e-3
C_POS, C_NEG, MUTED = "#2a78d6", "#eb6834", "#6b6b68"


def load(base, spec):
    run, _, it = spec.partition(":")
    snaps = sorted(glob.glob(os.path.join(base, run, "snap", "iter*.npz")))
    if it:
        snaps = [s for s in snaps if s.endswith(f"iter{int(it):03d}.npz")]
    if not snaps:
        return None
    z = np.load(snaps[-1])
    q, w, qff = np.asarray(z["q"]), np.asarray(z["omega"]), float(z["qff"])
    iq = int(np.argmin(np.abs(q - qff)))
    m = re.search(r"P0p(\d+)", run)
    title = f"P = {float('0.' + m.group(1)):.2f}" if m else run
    print(f"{run}  it {int(snaps[-1][-7:-4])}  Q = {q[iq] / qff:.4f} qff")
    return title, w, np.asarray(z["ImInvGamma"], float)[iq], np.asarray(z["ReInvGamma"], float)[iq]


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", default=None,
                    help="default: per ogni P la prima fra " + ", ".join(FAMILIES))
    ap.add_argument("--base", default=os.path.join(HERE, "out", "cluster"))
    ap.add_argument("--wlim", type=float, default=0.15)
    ap.add_argument("--wmin", type=float, default=1e-4, help="|Omega| minimo nel log-log")
    ap.add_argument("--part", choices=("im", "re"), default="im")
    ap.add_argument("--ws", type=float, default=0.005, help="--part re: dove si misura il gradino di ImGamma^-1")
    ap.add_argument("--ncol", type=int, default=4)
    ap.add_argument("--out", default=None, help="default out/cluster/plots/{im,re}gamma_qff_grid.png")
    a = ap.parse_args(argv)
    if a.out is None:
        a.out = os.path.join(HERE, "out", "cluster", "plots", f"{a.part}gamma_qff_grid.png")
    if not a.runs:
        a.runs = default_runs(a.base)
    data = [d for d in (load(a.base, s) for s in a.runs) if d is not None]
    nrow = -(-len(data) // a.ncol)
    figs = []
    for kind in ("lin", "log"):
        fig, axs = plt.subplots(nrow, a.ncol, figsize=(4.0 * a.ncol, 3.0 * nrow), layout="constrained",
                                squeeze=False)
        for ax, (title, w, im, rel) in zip(axs.flat, data):
            if a.part == "re":
                d = (rel - rel[int(np.argmin(np.abs(w)))]) / DELTA
                if kind == "lin":
                    m = np.abs(w) <= a.wlim
                    ax.plot(w[m], d[m], "-o", color=C_POS, ms=1.8, lw=0.9)
                    ax.axvline(0, color=MUTED, lw=0.6)
                    ax.set_xlim(-a.wlim, a.wlim)
                else:
                    for sgn, col, lab in ((1, C_POS, "Ω > 0"), (-1, C_NEG, "Ω < 0")):
                        m = (sgn * w > a.wmin) & (sgn * w <= a.wlim)
                        ax.semilogx(np.abs(w[m]), d[m], "-o", color=col, ms=1.8, lw=0.9, label=lab)
                    step = (np.interp(a.ws, w, im) - np.interp(-a.ws, w, im)) / DELTA
                    y0 = 0.5 * (np.interp(0.01, w, d) + np.interp(-0.01, w, d))
                    xs = np.geomspace(a.wmin, a.wlim, 20)
                    ax.semilogx(xs, y0 - step / np.pi * np.log(xs / 0.01), "--", color=MUTED, lw=0.8,
                                label=f"KK del gradino, D = {step:.1f} δ")
                    ax.set_xlim(a.wmin, a.wlim)
                    ax.legend(fontsize=6.5, frameon=False, loc="lower left")
                ax.axhline(0, color=MUTED, lw=0.6)
            elif kind == "lin":
                m = np.abs(w) <= a.wlim
                ax.plot(w[m], im[m] / DELTA, "-o", color=C_POS, ms=1.8, lw=0.9)
                ax.axvline(0, color=MUTED, lw=0.6)
                ax.axhline(0, color=MUTED, lw=0.6)
                ax.set_xlim(-a.wlim, a.wlim)
            else:
                for sgn, col, lab in ((1, C_POS, "Ω > 0"), (-1, C_NEG, "Ω < 0")):
                    m = (sgn * w > a.wmin) & (sgn * w <= a.wlim) & (np.abs(im) > 0)
                    ax.loglog(np.abs(w[m]), np.abs(im[m]), "-o", color=col, ms=1.8, lw=0.9, label=lab)
                # rette guida ancorate al lato negativo a |Omega| = 0.01
                mn = (w < 0) & (np.abs(im) > 0)
                if mn.any():
                    y0 = np.abs(np.interp(-0.01, w[mn], im[mn]))
                    xs = np.array([a.wmin, a.wlim])
                    for p, ls in ((1 / 3, ":"), (1 / 2, "--"), (2 / 3, "-.")):
                        ax.loglog(xs, y0 * (xs / 0.01) ** p, ls, color=MUTED, lw=0.8,
                                  label=f"∝ |Ω|^{p:.2g}")
                ax.set_xlim(a.wmin, a.wlim)
                ax.legend(fontsize=6.5, frameon=False, loc="lower right")
            ax.set_title(title, fontsize=10)
        for ax in axs[-1]:
            ax.set_xlabel("Ω" if kind == "lin" else "|Ω|")
        for ax in axs[:, 0]:
            if a.part == "re":
                ax.set_ylabel("ReΓ⁻¹(qff, Ω) − ReΓ⁻¹(qff, 0)  [δ]")
            else:
                ax.set_ylabel("ImΓ⁻¹(qff, Ω)  [δ]" if kind == "lin" else "|ImΓ⁻¹(qff, Ω)|")
        for ax in list(axs.flat)[len(data):]:
            ax.set_visible(False)
        figs.append(fig)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    root, ext = os.path.splitext(a.out)
    for fig, path in zip(figs, (a.out, f"{root}_log{ext}")):
        fig.savefig(path, dpi=130)
        print("scritto", path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
