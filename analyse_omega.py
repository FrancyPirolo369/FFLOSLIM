#!/usr/bin/env python3
"""Quanto bene la griglia omega risolve lo stato legato a due corpi?

Il polo di vuoto della T-matrice sta a  omega_b(q) = q^2/2 - mu_up - mu_dn - eps0
(a q=0, con mu=(1.65,0.35) ed eps0=1, vale -3.0).  E' l'oggetto che DEFINISCE
l'accoppiamento: se la griglia omega non ha nodi li', Gamma e' interpolata
linearmente attraverso un polo.
"""
import glob, os, sys
import numpy as np

qff = np.sqrt(1.65) - np.sqrt(0.35)
MU  = 1.65 + 0.35
EPS0 = 1.0
WB   = -MU - EPS0          # stato legato a q=0
WTH  = -MU                 # soglia di scattering a q=0

rows = []
for d in sorted(glob.glob("out/om_*")):
    f = os.path.join(d, "pair", "pair_gamma_table.npz")
    if not os.path.exists(f):
        continue
    z = np.load(f, allow_pickle=True)
    q = np.asarray(z["q"]); w = np.asarray(z["omega"])
    i0 = int(np.argmin(abs(w))); iq = int(np.argmin(abs(q - qff)))
    ws = np.sort(w)

    def gap_at(target):
        j = int(np.searchsorted(ws, target))
        lo = ws[max(j - 1, 0)]; hi = ws[min(j, ws.size - 1)]
        return hi - lo, lo, hi

    g_b, lo_b, hi_b = gap_at(WB)
    g_t, _, _       = gap_at(WTH)
    n_near = int(np.count_nonzero(np.abs(w - WB) <= 0.5))

    occ = w <= 0.0
    ap  = np.asarray(z["A_pair"])
    c2  = -np.trapezoid(q * np.trapezoid(ap[:, occ], w[occ], axis=1), q) / (2*np.pi)
    # peso che Sigma pesca dalla regione mal risolta
    band = (w >= -5.5) & (w <= -2.0)
    wgt  = np.trapezoid(np.abs(ap[iq][band]), w[band]) if band.sum() > 1 else np.nan

    rows.append(dict(tag=os.path.basename(d)[3:], nw=w.size,
                     gap_b=g_b, lo=lo_b, hi=hi_b, n_near=n_near, gap_t=g_t,
                     re=float(np.asarray(z["ReInvGamma"])[iq, i0]),
                     c2=float(c2), wgt=float(wgt)))

if not rows:
    sys.exit("nessuna tabella trovata")
print(f"stato legato atteso a omega = {WB:+.3f}   soglia a {WTH:+.3f}\n")
print(f"{'run':14s} {'nw':>5s} {'buco@-3':>9s} {'nodi|w+3|<.5':>13s} "
      f"{'buco@-2':>9s} {'ReG^-1(qff,0)':>15s} {'Delta_inf^2':>12s} {'|A| in[-5.5,-2]':>16s}")
print("-" * 100)
for r in rows:
    print(f"{r['tag']:14s} {r['nw']:5d} {r['gap_b']:9.3f} {r['n_near']:13d} "
          f"{r['gap_t']:9.3f} {r['re']:15.8f} {r['c2']:12.6f} {r['wgt']:16.6f}")
print("-" * 100)
b = rows[0]
for r in rows[1:]:
    print(f"{r['tag']:14s} vs {b['tag']}: "
          f"ReG^-1 {100*(r['re']-b['re'])/abs(b['re']):+7.2f}%   "
          f"contact {100*(r['c2']-b['c2'])/abs(b['c2']):+7.2f}%   "
          f"peso {100*(r['wgt']-b['wgt'])/abs(b['wgt']):+7.2f}%")
