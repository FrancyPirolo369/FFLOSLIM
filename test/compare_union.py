#!/usr/bin/env python3
"""Confronto PROD_UNION contro controllo sulle stesse cube (tabella pair dell'iterazione che
parte dallo stesso seed).  Uso: python3 test/compare_union.py [--base out/cluster]"""
import argparse, os, re, math
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELTA = 1e-3
PAIRS = [(0.80, "P0p80_prod_union", 1, "P0p80_x75a0p3", 16),
         (0.85, "P0p85_prod_union", 1, "P0p85_x75a0p3", 17),
         (0.90, "P0p90_prod_union", 1, "P0p90_x75a0p3", 19)]


def load(base, run, it):
    z = np.load(os.path.join(base, run, "snap", f"iter{it:03d}.npz"))
    return {k: np.asarray(z[k]) for k in z.files}


def med_resid(w, y, hw=0.03):
    lo = np.searchsorted(w, w - hw); hi = np.searchsorted(w, w + hw, side="right")
    return y - np.array([np.median(y[a:b]) for a, b in zip(lo, hi)])


def q_noise(q, y):
    r = np.full(q.size, np.nan)
    for i in range(1, q.size - 1):
        t = (q[i] - q[i - 1]) / (q[i + 1] - q[i - 1]); r[i] = y[i] - ((1 - t) * y[i - 1] + t * y[i + 1])
    return r / np.sqrt(1.5)


def loop_row(base, run, it):
    for l in open(os.path.join(base, run, "loop.log")):
        m = re.search(rf"iter {it} PAIR: contact=([\d.]+).*shift=([-+\d.eE]+)(?:.*Q=([\d.]+))?", l)
        if m:
            return float(m[1]), -4 * math.pi * float(m[2]) - 0.5 * math.log(2), float(m[3]) if m[3] else float("nan")
    return (float("nan"),) * 3


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--base", default=os.path.join(HERE, "out", "cluster"))
    ap.add_argument("--out", default=os.path.join(HERE, "out", "cluster", "plots", "union_vs_control.png"))
    a = ap.parse_args()
    fig, axs = plt.subplots(3, len(PAIRS), figsize=(5.4 * len(PAIRS), 10), layout="constrained", squeeze=False)
    for c, (P, ru, iu, rc, ic) in enumerate(PAIRS):
        U, C = load(a.base, ru, iu), load(a.base, rc, ic)
        assert np.allclose(U["q"], C["q"]) and np.allclose(U["omega"], C["omega"]), "griglie diverse"
        q, w, qff = U["q"], U["omega"], float(U["qff"])
        iw = int(np.argmin(np.abs(w))); x = q / qff
        print(f"== P = {P}: {ru} it {iu}  contro  {rc} it {ic}  (stesse cube)")
        res = {}
        for lab, T in (("union", U), ("controllo", C)):
            re0 = T["ReInvGamma"].astype(float)[:, iw]; im = T["ImInvGamma"].astype(float)
            win = x <= 2.0
            imax = int(np.nanargmax(np.where(win, re0, -np.inf)))
            neg = (w < -0.1) & (w > -2.5)
            rough = np.sqrt(np.nanmean([np.mean((med_resid(w[neg], im[i, neg]) / (np.abs(im[i, neg]) + 2e-3)) ** 2)
                                        for i in np.flatnonzero(x < 2)]))
            qn = q_noise(q, re0) / DELTA
            Cc, g, Qs = loop_row(a.base, ru if lab == "union" else rc, iu if lab == "union" else ic)
            res[lab] = dict(re0=re0, im=im, imax=imax)
            print(f"   {lab:9s} strisce ImG^-1 (rms rel, Omega<0) {rough*100:5.2f}% | rumore ReG^-1(Q,0) lungo Q [delta] "
                  f"Q<0.9qff {np.sqrt(np.nanmean(qn[x<0.9]**2)):.2f}  0.9-1.1 {np.sqrt(np.nanmean(qn[(x>=0.9)&(x<1.1)]**2)):.2f}  "
                  f"1.1-2 {np.sqrt(np.nanmean(qn[(x>=1.1)&(x<2)]**2)):.2f} | max a Q={x[imax]:.3f}qff, Q=0 - max = "
                  f"{(re0[0]-re0[imax])/DELTA:+.2f} delta | loop: C {Cc:.3f} g {g:+.4f}")
        d = (res["union"]["re0"] - res["controllo"]["re0"]) / DELTA
        dI = res["union"]["im"] - res["controllo"]["im"]
        print(f"   union - controllo: ReG^-1(Q,0) media {np.mean(d[x<2]):+.2f} delta (min {d[x<2].min():+.2f}, max {d[x<2].max():+.2f}); "
              f"max: {(res['union']['re0'][res['union']['imax']]-res['controllo']['re0'][res['controllo']['imax']])/DELTA:+.2f} delta; "
              f"ImG^-1 medio su Omega in [-2.5,-0.1]: {np.mean(dI[np.ix_(x<2,(w<-0.1)&(w>-2.5))])*1e3:+.3f}e-3")
        m = x <= 2
        for lab, col in (("controllo", "#2a78d6"), ("union", "#eb6834")):
            r = res[lab]["re0"]
            axs[0, c].plot(x[m], (r[m] - r[res[lab]["imax"]]) / DELTA, "-o", ms=2.5, lw=1, color=col, label=lab)
            for row, qv in ((1, 0.0), (2, qff)):
                i = int(np.argmin(np.abs(q - qv))); mw = (w > -2) & (w < 1)
                axs[row, c].plot(w[mw], res[lab]["im"][i, mw] * 1e3, lw=0.9, color=col, label=lab)
        axs[0, c].set(title=f"P={P}: ReΓ⁻¹(Q,0) − max [δ]", xlabel="Q/qff")
        axs[1, c].set(title=f"P={P}: ImΓ⁻¹(Q=0, Ω) ×10³", xlabel="Ω")
        axs[2, c].set(title=f"P={P}: ImΓ⁻¹(Q=qff, Ω) ×10³", xlabel="Ω")
        for r in range(3): axs[r, c].legend(fontsize=7, frameon=False)
    fig.savefig(a.out, dpi=120); print("scritto", a.out)


if __name__ == "__main__":
    main()
