#!/usr/bin/env python3
"""Grafici dalle istantanee di test/extract_snapshot.py (out/<base>/<RUN>/snap/).

Uso:  python3 test/plot_snapshots.py [BASE]      (default BASE = out/cluster)
Scrive in BASE/plots/:
  overview.png            confronto fra run in funzione dell'iterazione
  <RUN>.png               ReGamma^-1(Q,0) grezzo e shiftato, ImGamma^-1(qff,w), n(k)k^4
  <RUN>_imgamma.png       ImGamma^-1(Q,w) a Q = 0, qff, 1, 2, 4
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

DELTA = 1e-3
# shift congelati via variabile d'ambiente (motore vecchio, niente frozen_shift.txt)
FROZEN_ENV = {"p6_gfix": -0.019825}

# palette di riferimento (dataviz references/palette.md), ordine fisso
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARK = ["o", "s", "^", "D", "v", "P", "X", "*"]
RUNS = [  # (dir, etichetta)
    ("B_good_p6", "B  pair buona, P6, shift ricalcolato"),
    ("C_good_p6_frz", "C  pair buona, P6, shift congelato"),
    ("D_good_p12", "D  pair buona, P12"),
    ("F_good_p12_L6fix", "F  come D, Lambda6 + fix"),
    ("p6_gfix", "p6_gfix  turbo, P6, congelato"),
    ("A_turbo_p6", "A  turbo, P6, ricalcolato"),
]
# rampa sequenziale blu, step 250 -> 700 (ordinale: la piu' chiara resta >= 2:1)
RAMP = LinearSegmentedColormap.from_list(
    "iters", ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"])
INK, MUTED, GRID = "#1f1f1e", "#6b6b68", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "legend.frameon": False, "lines.linewidth": 1.4,
})


def load(base, run):
    out = []
    for f in sorted(glob.glob(os.path.join(base, run, "snap", "iter*.npz"))):
        z = dict(np.load(f))
        z["it"] = int(os.path.basename(f)[4:7])
        if run in FROZEN_ENV:
            z["shift_used"] = np.array(FROZEN_ENV[run])
        out.append(z)
    return out


def w0(z):
    return int(np.argmin(np.abs(z["omega"])))


def overview(base, data, dst):
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
    for c, (run, lab) in enumerate(RUNS):
        S = data.get(run)
        if not S:
            continue
        it = [z["it"] for z in S]
        C = np.array([float(z["pair_contact"]) for z in S])
        sh = np.array([float(z["shift_used"]) for z in S])
        eps = np.exp(-8 * np.pi * sh)
        dist = np.array([(float(z["reinv_qff_raw"]) - float(z["shift_used"])) / DELTA - 1
                         for z in S])
        tan = []
        for z in S:
            k = z["k"]
            tan.append(np.interp(4.0, k, z["n_dn"] * k ** 4) / float(z["pair_contact"]))
        kw = dict(color=CAT[c], marker=MARK[c], ms=4, label=lab)
        ax[0, 0].plot(it, C, **kw)
        ax[0, 1].plot(it, C / eps, **kw)
        ax[1, 0].plot(it, eps, **kw)
        ax[1, 1].plot(it, dist, **kw)
    ax[0, 0].set(title="contact C (dalla coppia)", xlabel="iterazione", ylabel="C")
    ax[0, 1].set(title="C / eps0_eff,  eps0_eff = exp(-8 pi shift)", xlabel="iterazione",
                 ylabel="C / eps0_eff")
    ax[0, 1].axhline(0.263, color=MUTED, ls=":", lw=1)
    ax[0, 1].annotate("0.263", (1, 0.263), xytext=(2, 3), textcoords="offset points",
                      color=MUTED)
    ax[1, 0].set(title="accoppiamento applicato  eps0_eff", xlabel="iterazione",
                 ylabel="eps0_eff  (eps0 nudo = 1)")
    ax[1, 1].set(title="ReGamma^-1_shift(qff, 0) / delta  (-1 = sulla soglia)",
                 xlabel="iterazione", ylabel="in unita' di delta = 1e-3")
    ax[1, 1].axhline(-1, color=MUTED, ls=":", lw=1)
    ax[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle("P = 0.65, mu fisso: confronto fra run", color=INK)
    fig.tight_layout()
    fig.savefig(dst, dpi=150)
    plt.close(fig)


def per_run(run, lab, S, dst):
    n = len(S)
    cols = [RAMP(i / max(n - 1, 1)) for i in range(n)]
    qff = float(S[0]["qff"])
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5), layout="constrained")
    for z, col in zip(S, cols):
        q, iw = z["q"], w0(z)
        r = z["ReInvGamma"][:, iw].astype(float)
        m = q <= 4.0
        ax[0, 0].plot(q[m], r[m], color=col)
        rs = r - float(z["shift_used"])
        m2 = q <= 2.0
        ax[0, 1].plot(q[m2], rs[m2] / DELTA, color=col, marker=".", ms=3)
        w = z["omega"]
        iq = int(np.argmin(np.abs(q - qff)))
        mw = (w > -4) & (w < 4)
        ax[1, 0].plot(w[mw], z["ImInvGamma"][iq, mw], color=col)
        k = z["k"]
        mk = (k >= 1.0) & (k <= 14)
        ax[1, 1].plot(k[mk], (z["n_dn"] * k ** 4)[mk], color=col)
    last = S[-1]
    k = last["k"]
    mk = (k >= 1.6) & (k <= 14)     # oltre kF_up = 1.28: sotto e' solo il gradino di Fermi
    ax[1, 1].plot(k[mk], (last["n_up"] * k ** 4)[mk], color=CAT[1], ls="--",
                  label=f"spin up, it {last['it']}")
    ax[1, 1].axhline(float(last["pair_contact"]), color=MUTED, ls=":", lw=1)
    ax[1, 1].annotate(f"C = {float(last['pair_contact']):.3f} (it {last['it']})",
                      (0.3, float(last["pair_contact"])), xytext=(0, 3),
                      textcoords="offset points", color=MUTED)
    for a in (ax[0, 0], ax[0, 1]):
        a.axvline(qff, color=MUTED, ls=":", lw=1)
    ax[0, 1].axhline(-1, color=MUTED, ls=":", lw=1)
    ax[0, 0].set(title="ReGamma^-1(Q, w=0) grezzo", xlabel="Q", ylabel="ReGamma^-1")
    ax[0, 1].set(title="ReGamma^-1(Q,0) - shift applicato  [delta = 1e-3]", xlabel="Q",
                 ylabel="in unita' di delta", ylim=(-15, 2))
    ax[1, 0].set(title=f"ImGamma^-1(qff = {qff:.3f}, w)", xlabel="w", ylabel="ImGamma^-1")
    ax[1, 1].set(title="n(k) k^4  spin down (rampa, k>=1) e up (tratteggio, k>=1.6)",
                 xlabel="k", ylabel="n(k) k^4",
                 ylim=(0, 1.4 * float(last["pair_contact"])))
    ax[1, 1].legend(loc="upper right", fontsize=8)
    sm = plt.cm.ScalarMappable(cmap=RAMP, norm=plt.Normalize(S[0]["it"], S[-1]["it"]))
    fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.01, label="iterazione")
    fig.suptitle(lab, color=INK)
    fig.savefig(dst, dpi=150)
    plt.close(fig)


def imgamma(run, lab, S, dst):
    qff = float(S[0]["qff"])
    Qs = [0.0, qff, 1.0, 2.0, 4.0]
    pick = sorted({0, len(S) // 2, len(S) - 1})
    fig, ax = plt.subplots(1, len(Qs), figsize=(15, 3.6))
    for j in pick:
        z = S[j]
        col = RAMP(j / max(len(S) - 1, 1))
        q, w = z["q"], z["omega"]
        for a, Q in zip(ax, Qs):
            iq = int(np.argmin(np.abs(q - Q)))
            th = q[iq] ** 2 / 2 - 2.0      # soglia libera Q^2/2 - mu_up - mu_dn (mu etichetta)
            lo, hi = min(-4.0, th - 4.0), max(4.0, th + 8.0)
            m = (w > lo) & (w < hi)
            a.plot(w[m], z["ImInvGamma"][iq, m], color=col, label=f"it {z['it']}")
            a.set_title(f"Q = {q[iq]:.3f}")
            a.axvline(th, color=MUTED, ls=":", lw=1)
    ax[0].set_ylabel("ImGamma^-1(Q, w)")
    for a in ax:
        a.set_xlabel("w")
    ax[-1].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"{lab}  (linea punteggiata: soglia libera Q^2/2 - mu_up - mu_dn)",
                 color=INK)
    fig.tight_layout()
    fig.savefig(dst, dpi=150)
    plt.close(fig)


def main(argv):
    base = argv[0] if argv else os.path.join("out", "cluster")
    outd = os.path.join(base, "plots")
    os.makedirs(outd, exist_ok=True)
    data = {run: load(base, run) for run, _ in RUNS}
    overview(base, data, os.path.join(outd, "overview.png"))
    for run, lab in RUNS:
        if data[run]:
            per_run(run, lab, data[run], os.path.join(outd, f"{run}.png"))
            imgamma(run, lab, data[run], os.path.join(outd, f"{run}_imgamma.png"))
    print("scritti in", outd, sorted(os.listdir(outd)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
