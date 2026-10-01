#!/usr/bin/env python3
"""Segni di criticita' lungo la linea FFLO, dall'ultima snapshot di ogni run.

Per ogni P (default: prod fino a 0.65, gmax a 0.70):
  anello    ReGamma^-1(Q, 0) pinnata (max = -delta) attorno a Q*: fit a + b1 x + b2 x^2 con
            x = (Q - Q*)/qff su |x| < 0.12; curvatura b2 [delta] e rugosita' (RMS del residuo) [delta]
  modo      ImGamma^-1(Q*, Omega): esponente beta (pendenza log-log) su [3e-3, 1e-2] e [1e-2, 3e-2],
            per Omega > 0 e < 0, e nodi azzerati dal sign-guard entro |Omega| < 0.05
  fermioni  ImSigma(kF, omega) di ciascuno spin (righe salvate in sigmaNNN.npz): esponente alpha su
            [7e-3, 3e-2] e [3e-2, 0.1] (media dei due lati), ImSigma(kF, 0) (a T = 0 deve essere 0),
            Z(omega) = 1/(1 - [ReS(w) - ReS(-w)]/2w) a omega = 0.007 e 0.1
  relazione beta atteso dalla tangenza con minoritario non-FL: 1 - alpha_dn / 2 (liquido di Fermi: 1/2)
Figura: 2 x 3 pannelli, colore = P su una scala sequenziale.

Uso (dalla radice di SLIM):  python3 test/critical_hallmarks.py [RUN ...] [--out out/cluster/plots/critical_hallmarks.png]
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
DELTA = 1e-3
INK, MUTED, GRID, SURF = "#1f1f1e", "#6b6b68", "#e4e3df", "#fcfcfb"
DEFAULT = ["P0p10_prod", "P0p20_prod", "P0p30_prod", "P0p40_prod", "P0p50_prod", "P0p60_prod",
           "P0p65_prod", "P0p70_gmax"]
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False, "font.size": 8.5,
    "legend.frameon": False, "lines.linewidth": 1.4,
})


def slope(x, y, lo, hi):
    m = (np.abs(x) >= lo) & (np.abs(x) <= hi) & (np.abs(y) > 0) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    return float(np.polyfit(np.log(np.abs(x[m])), np.log(np.abs(y[m])), 1)[0])


def sigma_row(s, spin):
    mu = float(s[f"mu_{'up' if spin == 'up' else 'dn'}_{spin}"])
    kf = np.sqrt(mu)
    rk = np.asarray(s[f"rows_k_{spin}"], float)
    i = int(np.argmin(np.abs(rk - kf)))
    w = np.asarray(s[f"rows_w_{spin}"][i], float)
    o = np.argsort(w)
    return w[o], np.asarray(s[f"rows_ReS_{spin}"][i], float)[o], np.asarray(s[f"rows_ImS_{spin}"][i], float)[o]


def zeta(w, re, x):
    return 1.0 / (1.0 - (np.interp(x, w, re) - np.interp(-x, w, re)) / (2 * x))


def analyse(base, run):
    snaps = sorted(glob.glob(os.path.join(base, run, "snap", "iter*.npz")))
    if not snaps:
        return None
    z = np.load(snaps[-1])
    it = snaps[-1][-7:-4]
    q, w, qff = np.asarray(z["q"], float), np.asarray(z["omega"], float), float(z["qff"])
    re, im = np.asarray(z["ReInvGamma"], float), np.asarray(z["ImInvGamma"], float)
    i0 = int(np.argmin(np.abs(w)))
    iq = int(np.argmax(np.where(q <= 2 * qff, re[:, i0], -np.inf)))
    pinned = re[:, i0] - re[iq, i0] - DELTA
    x = (q - q[iq]) / qff
    m = np.abs(x) < 0.12
    coef = np.polyfit(x[m], pinned[m], 2)
    resid = pinned[m] - np.polyval(coef, x[m])
    imq = im[iq]
    near = (np.abs(w) <= 0.05) & (w != 0)
    out = dict(run=run, it=it, P=float("0." + re_p(run)), qstar=q[iq] / qff, x=x, pinned=pinned, coef=coef,
               b2=-coef[0] / DELTA, rough=float(np.sqrt(np.mean(resid ** 2))) / DELTA, w=w, imq=imq,
               zeros=int(np.sum(imq[near] == 0)))
    for sg, lab in ((1, "p"), (-1, "m")):
        mm = sg * w > 0
        out[f"beta1_{lab}"] = slope(w[mm], imq[mm], 3e-3, 1e-2)
        out[f"beta2_{lab}"] = slope(w[mm], imq[mm], 1e-2, 3e-2)
    ss = sorted(glob.glob(os.path.join(base, run, "snap", "sigma*.npz")))
    if ss:
        s = np.load(ss[-1])
        for spin in ("dn", "up"):
            sw, sre, sim = sigma_row(s, spin)
            out[f"sig_{spin}"] = (sw, sre, sim)
            a1 = [slope(sw[sg * sw > 0], sim[sg * sw > 0], 7e-3, 3e-2) for sg in (1, -1)]
            a2 = [slope(sw[sg * sw > 0], sim[sg * sw > 0], 3e-2, 0.1) for sg in (1, -1)]
            out[f"alpha1_{spin}"], out[f"alpha2_{spin}"] = float(np.nanmean(a1)), float(np.nanmean(a2))
            out[f"im0_{spin}"] = float(np.interp(0.0, sw, sim))
            out[f"z007_{spin}"], out[f"z1_{spin}"] = zeta(sw, sre, 0.007), zeta(sw, sre, 0.1)
    return out


def re_p(run):
    m = re.search(r"P0p(\d+)", run)
    return m.group(1) if m else "0"


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", default=DEFAULT)
    ap.add_argument("--base", default=os.path.join(HERE, "out", "cluster"))
    ap.add_argument("--out", default=os.path.join(HERE, "out", "cluster", "plots", "critical_hallmarks.png"))
    a = ap.parse_args(argv)
    res = [r for r in (analyse(a.base, run) for run in a.runs) if r is not None]
    print(f"{'run':16s} it  | {'Q*/qff':>6s} {'b2[d]':>7s} {'rug[d]':>6s} | {'beta+ 3e-3':>10s} {'1e-2':>5s} "
          f"{'beta- 3e-3':>10s} {'1e-2':>5s} {'zeri':>4s} | {'a_dn':>5s} {'a_dn2':>5s} {'1-a/2':>5s} "
          f"{'ImS_dn(0)':>9s} {'Z_dn .007/.1':>12s} | {'a_up':>5s} {'a_up2':>5s} {'Z_up .007/.1':>12s}")
    for r in res:
        print(f"{r['run']:16s} {r['it']} | {r['qstar']:6.3f} {r['b2']:7.0f} {r['rough']:6.2f} | "
              f"{r['beta1_p']:10.2f} {r['beta2_p']:5.2f} {r['beta1_m']:10.2f} {r['beta2_m']:5.2f} {r['zeros']:4d} | "
              f"{r.get('alpha1_dn', np.nan):5.2f} {r.get('alpha2_dn', np.nan):5.2f} "
              f"{1 - r.get('alpha1_dn', np.nan) / 2:5.2f} {r.get('im0_dn', np.nan):9.1e} "
              f"{r.get('z007_dn', np.nan):5.3f}/{r.get('z1_dn', np.nan):5.3f} | "
              f"{r.get('alpha1_up', np.nan):5.2f} {r.get('alpha2_up', np.nan):5.2f} "
              f"{r.get('z007_up', np.nan):5.3f}/{r.get('z1_up', np.nan):5.3f}")

    cmap = plt.get_cmap("Blues")
    cols = {r["run"]: cmap(0.35 + 0.65 * i / max(1, len(res) - 1)) for i, r in enumerate(res)}
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.5), layout="constrained")
    for r in res:
        c, lab = cols[r["run"]], f"P = {r['P']:.2f}"
        m = np.abs(r["x"]) < 0.3
        axs[0, 0].plot(r["x"][m], r["pinned"][m] / DELTA, "o", ms=2.5, color=c, label=lab)
        xs = np.linspace(-0.12, 0.12, 50)
        axs[0, 0].plot(xs, np.polyval(r["coef"], xs) / DELTA, "-", lw=0.9, color=c)
        for col, sg in ((1, 1), (2, -1)):
            mm = sg * r["w"] > 0
            y = sg * r["imq"][mm]
            axs[0, col].loglog(np.abs(r["w"][mm]), np.where(y > 0, y / DELTA, np.nan), "-", color=c, label=lab)
        if "sig_dn" in r:
            for col, spin in ((0, "dn"), (1, "up")):
                sw, sre, sim = r[f"sig_{spin}"]
                mm = sw < 0
                axs[1, col].loglog(-sw[mm], np.abs(sim[mm]), "-", color=c, label=lab)
            sw, sre, _ = r["sig_dn"]
            xs = np.geomspace(0.007, 0.25, 12)
            axs[1, 2].semilogx(xs, [zeta(sw, sre, x) for x in xs], "-o", ms=2.5, color=c, label=lab)
    axs[0, 0].set(xlabel="(Q − Q*) / qff", ylabel="ReΓ⁻¹(Q, 0) pinnata  [δ]", title="anello: ReΓ⁻¹ attorno a Q* (fit parabolico)",
                  xlim=(-0.3, 0.3), ylim=(-60, 5))
    for col, t in ((1, "Ω > 0"), (2, "Ω < 0")):
        ax = axs[0, col]
        xs = np.array([1e-4, 0.1])
        ax.loglog(xs, 2.0 * (xs / 1e-3) ** 0.5, ":", color=MUTED, lw=0.9, label="∝ |Ω|^½ (Fermi liquido)")
        ax.loglog(xs, 2.0 * (xs / 1e-3) ** (2 / 3), "--", color=MUTED, lw=0.9, label="∝ |Ω|^⅔")
        ax.set(xlabel="|Ω|", ylabel="|ImΓ⁻¹(Q*, Ω)|  [δ]", title=f"modo di coppia a Q*, {t}", xlim=(1e-4, 0.3), ylim=(1e-2, 50))
    for col, spin in ((0, "↓ (minoritario)"), (1, "↑ (maggioritario)")):
        ax = axs[1, col]
        xs = np.array([3e-3, 0.3])
        for p_, ls, lab in ((2 / 3, "--", "∝ |ω|^⅔"), (1.0, "-.", "∝ |ω|"), (2.0, ":", "∝ ω² (Fermi liquido)")):
            ax.loglog(xs, 0.05 * (xs / 0.03) ** p_, ls, color=MUTED, lw=0.9, label=lab)
        ax.axvline(0.007, color=MUTED, lw=0.6)
        ax.set(xlabel="|ω|, ω < 0", ylabel="|ImΣ(kF, ω)|", title=f"fermioni {spin} a kF (sotto 7e-3: nodi di Σ)", xlim=(1e-3, 1), ylim=(3e-4, 1))
    axs[1, 2].set(xlabel="scala ω", ylabel="Z↓(ω) a kF↓", title="Z del minoritario per scala")
    for ax in axs.flat:
        ax.legend(fontsize=6.5, ncol=2)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=115)
    print("scritto", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
