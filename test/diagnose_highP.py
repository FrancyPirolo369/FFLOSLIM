#!/usr/bin/env python3
"""Cosa va storto ad alta P: canali supercritici, contact per Q, sigma0, n(k).

Uso (dalla radice):  python3 test/diagnose_highP.py RUN [RUN ...] [--base out/cluster]
Legge out/cluster/<RUN>/snap/iterNNN.npz e sigmaNNN.npz (extract_snapshot/extract_sigma).
Stampa per ogni iterazione: C, shift, massimo di ReGamma^-1(Q,0) dopo lo shift (in delta,
> 0 = canale oltre la criticita') e dove sta, frazione del contact per fasce di Q,
sigma0 dei due spin, densita' fresche.  Scrive out/cluster/plots/<RUN>_highP.png.
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
sys.path.insert(0, HERE)
DELTA = 1.0e-3
INK, MUTED, GRID = "#1f1f1e", "#6b6b68", "#e4e3df"
RAMP = LinearSegmentedColormap.from_list(
    "iters", ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"])
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "legend.frameon": False, "lines.linewidth": 1.2,
})


def shifted_pair(z):
    from fflo.pair_shifted import apply_shift_to_pair_table
    q = np.asarray(z["q"], float)
    t = {"q": q, "omega": np.asarray(z["omega"], float),
         "ReInvGamma": np.asarray(z["ReInvGamma"], float),
         "ImInvGamma": np.asarray(z["ImInvGamma"], float)}
    qff = float(z["qff"])
    iq = int(np.argmin(np.abs(q - qff)))
    return apply_shift_to_pair_table(t, float(q[iq]), subcritical_delta=DELTA,
                                     eta_floor_mode="exact_zero",
                                     base_shift_override=float(z["shift_used"]))


def analyse(z):
    s = shifted_pair(z)
    q, w = np.asarray(s["q"]), np.asarray(s["omega"])
    iw = int(np.argmin(np.abs(w)))
    re0 = np.asarray(s["ReInvGamma"])[:, iw]
    m = q <= 4.0
    j = int(np.argmax(np.where(m, re0, -np.inf)))
    A = np.asarray(s["A_pair"])
    occ = w <= 0.0
    per_q = -np.trapezoid(A[:, occ], w[occ], axis=1) * q / (2 * np.pi) / 4.0   # integrando del contact
    # poli stretti (non smorzati o quasi): radici di ReGamma^-1(Q,w) con larghezza
    # |ImGamma^-1|/|pendenza| < 0.05.  Peso vero Z = 1/|pendenza| (paper, Eq. 17), contro il
    # peso che la griglia cattura integrando A_pair in una finestra attorno.  Pesati in Q
    # come il contact (Q dQ/2pi/4).
    reS, imS = np.asarray(s["ReInvGamma"]), np.asarray(s["ImInvGamma"])
    dq = np.gradient(q)
    pole = {"neg_true": 0.0, "neg_got": 0.0, "pos_true": 0.0, "pos_got": 0.0}
    sel = np.flatnonzero(np.abs(w) < 8.0)
    for i in range(q.size):
        if q[i] > 4.0:
            continue
        re, im = reS[i, sel], imS[i, sel]
        for c in np.flatnonzero(np.diff(np.sign(re)) != 0):
            a0, b0 = sel[c], sel[c] + 1
            slope = (reS[i, b0] - reS[i, a0]) / (w[b0] - w[a0])
            if slope == 0.0:
                continue
            ws = w[a0] - reS[i, a0] / slope
            width = abs(np.interp(ws, w, imS[i])) / abs(slope)
            if width > 0.05:
                continue
            h = max(3 * width, 0.02)
            win = (w > ws - h) & (w < ws + h)
            got = abs(np.trapezoid(A[i, win], w[win])) if win.sum() > 1 else 0.0
            wt = q[i] * dq[i] / (2 * np.pi) / 4.0
            key = "neg" if ws < 0 else "pos"
            pole[key + "_true"] += wt / abs(slope)
            pole[key + "_got"] += wt * got
    return dict(q=q, re0=re0, qmax=q[j], remax=re0[j], per_q=per_q, pole=pole)


def band(q, f, a, b):
    m = (q >= a) & (q <= b)
    return np.trapezoid(f[m], q[m]) if m.sum() > 1 else 0.0


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--base", default=os.path.join(HERE, "out", "cluster"))
    a = ap.parse_args(argv)
    for run in a.runs:
        F = sorted(glob.glob(os.path.join(a.base, run, "snap", "iter*.npz")))
        if not F:
            print(f"{run}: nessuna istantanea"); continue
        z0 = np.load(F[0])
        qff = float(z0["qff"])
        mu_dn = float(np.sqrt(0) + 0)
        print(f"\n=== {run}  (qff = {qff:.4f}) ===")
        print(" it     C      shift   max ReG^-1_shift(Q,0) [delta] a Q   | contact da Q<qff/2  ~qff  >2qff |"
              " sigma0 dn / up   | n fresca dn / up")
        rows = []
        for f in F:
            z = np.load(f)
            it = int(os.path.basename(f)[4:7])
            r = analyse(z)
            q, pq = r["q"], r["per_q"]
            tot = np.trapezoid(pq, q)
            fr = (band(q, pq, 0, qff / 2) / tot, band(q, pq, qff / 2, 2 * qff) / tot,
                  band(q, pq, 2 * qff, q[-1]) / tot)
            sg = f.replace("iter", "sigma")
            s0d = s0u = fd = fu = float("nan")
            if os.path.exists(sg):
                g = np.load(sg)
                s0d, s0u = float(g["sigma0_dn"]), float(g["sigma0_up"])
                mud, muu = float(g["mu_dn_dn"]), float(g["mu_up_up"])
                fd = float(g["dr_alpha_1_density_down"]) / (mud / 2) - 1
                fu = float(g["dr_alpha_1_density_up"]) / (muu / 2) - 1
            rows.append((it, r, z))
            pl = r["pole"]
            print(f" {it:2d}  {float(z['pair_contact']):6.3f}  {float(z['shift_used']):+.4f}   "
                  f"{r['remax'] / DELTA:+9.1f} a Q={r['qmax']:.3f}          |"
                  f"   {fr[0]:5.1%}   {fr[1]:5.1%}  {fr[2]:5.1%} |"
                  f"  {s0d:+.3f} / {s0u:+.3f} | {fd:+7.1%} / {fu:+6.1%} |"
                  f" poli stretti w<0 vero/preso {pl['neg_true']:.3f}/{pl['neg_got']:.3f}"
                  f"  w>0 {pl['pos_true']:.3f}/{pl['pos_got']:.3f}")
        # grafico: profilo, contact per Q, n_dn(k), k^2 n_dn
        n = len(rows)
        fig, ax = plt.subplots(1, 4, figsize=(18, 4.2), layout="constrained")
        for idx, (it, r, z) in enumerate(rows):
            col = RAMP(idx / max(n - 1, 1))
            m = r["q"] <= 3.0
            ax[0].plot(r["q"][m], r["re0"][m] / DELTA, color=col)
            ax[1].plot(r["q"][m], r["per_q"][m], color=col)
            k, nd = z["k_dn"], z["n_dn"]
            mk = k <= 2.5
            ax[2].plot(k[mk], nd[mk], color=col)
            mk = (k > 0.05) & (k <= 14)
            ax[3].plot(k[mk], (k * k * nd)[mk], color=col)
        for x in ax[:2]:
            x.axvline(qff, color=MUTED, ls=":", lw=1)
        ax[0].axhline(0, color="#e34948", lw=0.8)
        ax[0].set(title="ReGamma^-1(Q,0) dopo lo shift [delta]  (> 0: oltre la criticita')",
                  xlabel="Q", ylim=(-60, 30))
        ax[1].set(title="contact: integrando in Q", xlabel="Q")
        ax[2].set(title="n_down(k)", xlabel="k")
        ax[3].set(title="k^2 n_down(k)  (area in ln k = densita')", xlabel="k", xscale="log")
        sm = plt.cm.ScalarMappable(cmap=RAMP, norm=plt.Normalize(rows[0][0], rows[-1][0]))
        fig.colorbar(sm, ax=ax, fraction=0.012, pad=0.01, label="iterazione")
        fig.suptitle(f"{run}", color=INK)
        outd = os.path.join(a.base, "plots")
        os.makedirs(outd, exist_ok=True)
        fig.savefig(os.path.join(outd, f"{run}_highP.png"), dpi=140)
        print("scritto", os.path.join(outd, f"{run}_highP.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
