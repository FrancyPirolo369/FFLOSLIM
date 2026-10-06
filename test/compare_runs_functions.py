#!/usr/bin/env python3
"""Confronto delle funzioni fra stati di run diverse (snapshot di pipeline), una figura 3x3.

Ogni stato e' RUN:IT (per esempio P0p50_prod:30).  Dalle snapshot di out/cluster/RUN/snap:
    iterIT.npz   ReGamma^-1, ImGamma^-1 (grezze, senza pin), n(k), contatto, shift
    sigmaIT.npz  righe di Sigma (rows_*), ReSigma(k, 0) sul reticolo k, nodi k di Sigma (ck_*_k_sparse)
e, se c'e', dalla cube iterIT/next_cubes la N(omega) (k < 3, Gamma visiva 0.002).

Pannelli:  Gamma:  ReG^-1(Q,0) - ReG^-1(qff,0)  |  ImG^-1(qff, Omega) vicino a 0  |  ImG^-1(qff, Omega>0) log-log
           Sigma:  -ImSigma_dn(kF_dn, w)  |  -ImSigma_up(kF_up, w)  |  ReSigma(k, 0) - sigma0 (con i nodi k)
           oss.:   n(k)  |  n(k) k^4  |  N(omega)

Uso (dalla radice di SLIM):
    python3 test/compare_runs_functions.py P0p50_prod:30 P0p50_final:1 P0p50_final:6 --out out/cluster/plots/x.png
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G0 = -0.5 * math.log(2.0)
COLORS = ("#eb6834", "#7fb2ea", "#1f5fae", "#3a9e5c", "#8a5cc2")
STYLES = ("-", "--", "-", "-", "-")


def load_state(base, spec):
    run, it = spec.split(":")
    it = int(it)
    rd = os.path.join(base, run)
    z = np.load(os.path.join(rd, "snap", f"iter{it:03d}.npz"), allow_pickle=True)
    s = np.load(os.path.join(rd, "snap", f"sigma{it:03d}.npz"), allow_pickle=True)
    cubes = {}
    for sp in ("up", "down"):
        f = glob.glob(os.path.join(rd, f"iter{it:03d}", "next_cubes", f"A_komega_spin{sp}_iter*.npz"))
        if f:
            cubes[sp] = f[0]
    shift = float(z["shift_used"])
    return dict(label=f"{run} it{it}", z=z, s=s, cubes=cubes,
                g=-4.0 * math.pi * shift + G0, C=float(z["pair_contact"]))


def dos(path, w, gam=0.002, kmax=3.0):
    c = np.load(path, allow_pickle=True)
    k, W, re, im = c["k"], c["w"], c["ReS"], c["ImS"]
    mu = float(c["mu_dn"] if "spindown" in path else c["mu_up"])
    s0, m = float(c["sigma0"]), float(c["mass"])
    sel = np.flatnonzero(k < kmax)
    A = np.empty((sel.size, w.size))
    for j, i in enumerate(sel):
        r = np.interp(w, W[i], re[i])
        g = np.abs(np.interp(w, W[i], im[i])) + gam
        A[j] = g / np.pi / ((w - (k[i] ** 2 / m - mu) - (r - s0)) ** 2 + g ** 2)
    return np.trapezoid(A * k[sel][:, None], k[sel], axis=0)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("states", nargs="+", help="RUN:IT")
    ap.add_argument("--base", default=os.path.join(HERE, "out", "cluster"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    st = [load_state(a.base, x) for x in a.states]

    fig, ax = plt.subplots(3, 3, figsize=(17, 13), layout="constrained")
    wd = np.linspace(-0.12, 0.12, 961)
    for n, S in enumerate(st):
        col, ls = COLORS[n % len(COLORS)], STYLES[n % len(STYLES)]
        lab = f"{S['label']}  (g = {S['g']:+.3f}, C = {S['C']:.3f})"
        z, s = S["z"], S["s"]
        q, om = np.asarray(z["q"], float), np.asarray(z["omega"], float)
        re, im = np.asarray(z["ReInvGamma"], float), np.asarray(z["ImInvGamma"], float)
        qff = float(z["qff"])
        i0, iq = int(np.argmin(np.abs(om))), int(np.argmin(np.abs(q - qff)))
        m = q > 1e-9
        ax[0, 0].plot(q[m] / qff, re[m, i0] - re[iq, i0], ls, color=col, lw=1.5, marker=".", ms=3, label=lab)
        mw = np.abs(om) <= 0.06
        ax[0, 1].plot(om[mw], im[iq, mw], ls, color=col, lw=1.5, label=lab)
        mp = (om > 5e-4) & (om < 3.0)
        ax[0, 2].loglog(om[mp], np.abs(im[iq, mp]), ls, color=col, lw=1.5, label=lab)

        for c_, tag in ((0, "dn"), (1, "up")):
            mu = float(s[f"mu_{tag}_{tag}"])
            kf = math.sqrt(mu)
            rk = np.asarray(s[f"rows_k_{tag}"], float)
            j = int(np.argmin(np.abs(rk - kf)))
            w_ = np.asarray(s[f"rows_w_{tag}"], float)[j]
            ims = np.asarray(s[f"rows_ImS_{tag}"], float)[j]
            mm = np.abs(w_) <= 0.3
            ax[1, c_].plot(w_[mm], -ims[mm], ls, color=col, lw=1.5, label=lab + f"  [riga k = {rk[j]:.3f}]")
            kk = np.asarray(s[f"k_{tag}"], float)
            r0 = np.asarray(s[f"reS0_{tag}"], float) - float(s[f"sigma0_{tag}"])
            mk = (kk / kf > 0.4) & (kk / kf < 1.8)
            ax[1, 2].plot(kk[mk] / kf, r0[mk], "-" if tag == "dn" else ":", color=col, lw=1.5,
                          label=f"{S['label']} {'dn' if tag == 'dn' else 'up'}")
            nodes = np.asarray(s[f"ck_{tag}_k_sparse"], float).ravel()
            nodes = nodes[(nodes / kf > 0.4) & (nodes / kf < 1.8)]
            ax[1, 2].plot(nodes / kf, np.interp(nodes, kk, r0), "o" if tag == "dn" else "s", color=col,
                          ms=5, mfc="none")

        for tag, kk_key, nk_key, sty in (("dn", "k_dn", "n_dn", "-"), ("up", "k", "n_up", ":")):
            kk, nk = np.asarray(z[kk_key], float), np.asarray(z[nk_key], float)
            mk = kk <= 2.0
            ax[2, 0].plot(kk[mk], nk[mk], sty, color=col, lw=1.5, label=f"{S['label']} {tag}")
            mk = (kk > 0.3) & (kk <= 12.0)
            ax[2, 1].plot(kk[mk], nk[mk] * kk[mk] ** 4, sty, color=col, lw=1.5, label=f"{S['label']} {tag}")

        for sp, sty in (("down", "-"), ("up", ":")):
            if sp in S["cubes"]:
                ax[2, 2].plot(wd, dos(S["cubes"][sp], wd), sty, color=col, lw=1.5,
                              label=f"{S['label']} {'dn' if sp == 'down' else 'up'}")

    ax[0, 0].axhline(0, color="0.5", lw=0.8)
    ax[0, 0].set(xlabel="Q / qff", ylabel="ReG^-1(Q,0) - ReG^-1(qff,0)", xlim=(0, 1.6), ylim=(-0.03, 0.004),
                 title="Gamma: profilo in Q a omega = 0 (0 = pin a qff)")
    ax[0, 1].axhline(0, color="0.5", lw=0.8)
    ax[0, 1].axvline(0, color="0.5", lw=0.8)
    ax[0, 1].set(xlabel="Omega", ylabel="ImG^-1(qff, Omega)", title="ImG^-1 a qff vicino a Omega = 0 (bolla)")
    ref = np.array([1e-3, 1e-1])
    for e, lsr in ((0.5, ":"), (2 / 3, "-.")):
        ax[0, 2].loglog(ref, 2e-2 * (ref / 1e-1) ** e, lsr, color="0.4", lw=1, label=f"Omega^{e:.2f} (guida)")
    ax[0, 2].set(xlabel="Omega", ylabel="|ImG^-1(qff, Omega)|", title="ImG^-1(qff, Omega > 0), log-log")
    for c_, sp in ((0, "dn"), (1, "up")):
        ax[1, c_].axvline(0, color="0.5", lw=0.8)
        ax[1, c_].set(xlabel="omega", ylabel=f"-ImSigma_{sp}(kF_{sp}, omega)",
                      title=f"Sigma {sp} alla riga piu' vicina a kF_{sp}")
    ax[1, 2].axvline(1, color="0.5", lw=0.8)
    ax[1, 2].set(xlabel="k / kF", ylabel="ReSigma(k, 0) - sigma0",
                 title="ReSigma(k,0): pieno dn, punti up; simboli = nodi k")
    ax[2, 0].set(xlabel="k", ylabel="n(k)", title="n(k): pieno = dn, punti = up")
    ax[2, 1].set(xlabel="k", ylabel="n(k) k^4", xscale="log", title="coda di contatto n(k) k^4")
    ax[2, 2].set(xlabel="omega", ylabel="N(omega)", title="DOS (dalle cube, Gamma visiva 0.002)")
    for x in ax.ravel():
        x.grid(alpha=0.3)
        x.legend(fontsize=7)
    fig.suptitle("  vs  ".join(S["label"] for S in st), fontsize=12)
    fig.savefig(a.out, dpi=100)
    print("scritto", a.out)
    for S in st:
        print(f"  {S['label']:22s} g = {S['g']:+.4f}  C = {S['C']:.4f}  N(omega) da cube: {sorted(S['cubes'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
