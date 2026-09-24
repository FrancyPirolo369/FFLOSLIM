#!/usr/bin/env python3
"""Passo 1 della verifica "Gamma e' il ladder di G?": i ritocchi a valle della bolla.

pair_gamma (proxy_residual) costruisce
    Gamma^-1 = Gamma_ref^-1 - [ KK(ImD*T) + i ImD*T ],   ImD = ImPi_num - ImPi_ref
con T il taper del residuo, poi applica la guardia di segno su ImGamma^-1; density
aggiunge il floor eta e lo shift.  A parita' di ImPi_num, il ladder esatto vuole
T = 1 e nessun ritocco.  Qui si rifa' il KK su ImD grezzo (ImDeltaPiResidualRaw)
sulla finestra della tabella e si misura, in unita' di delta, quanto cambia
Gamma^-1 attorno a (qff, |w| < 1) dopo aver tolto la costante in (qff, 0) che il
re-pinning dello shift assorbe comunque.

Uso:  python3 test/gamma_ladder_check.py PAIR_TABLE.npz [--delta 1e-3]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("table")
    ap.add_argument("--delta", type=float, default=1.0e-3)
    a = ap.parse_args()
    from fflo.kramers_kronig import kk_re_pv_linear

    z = np.load(a.table, allow_pickle=True)
    q, w = np.asarray(z["q"], float), np.asarray(z["omega"], float)
    mu_up = float(np.asarray(z["mu_up"]).reshape(-1)[0])
    mu_dn = float(np.asarray(z["mu_dn"]).reshape(-1)[0])
    qff = np.sqrt(mu_up) - np.sqrt(mu_dn)
    raw = np.asarray(z["ImDeltaPiResidualRaw"], float)
    tap = np.asarray(z["ImDeltaPiResidual"], float)
    re_tab = np.asarray(z["ReDeltaPiResidual"], float)
    d = a.delta

    # 0) il mio KK riproduce quello del pipeline sul residuo con taper?
    iq0 = int(np.argmin(np.abs(q - qff)))
    w0 = int(np.argmin(np.abs(w)))
    win = np.abs(w) < 1.0
    check = kk_re_pv_linear(w, tap[iq0])
    print(f"tabella: nQ={q.size} nW={w.size} w in [{w[0]:g},{w[-1]:g}]  qff={qff:.4f} (Q[{iq0}]={q[iq0]:.4f})")
    print(f"controllo KK (residuo con taper, Q=qff, |w|<1): max|mio - tabella| = "
          f"{np.abs(check - re_tab[iq0])[win].max() / d:.3f} delta")
    edge = max(abs(raw[iq0, 0]), abs(raw[iq0, -1]))
    print(f"residuo grezzo ai bordi della finestra (Q=qff): {edge:.2e}  "
          f"(taper: peso tolto = {np.trapezoid(np.abs(raw[iq0] - tap[iq0]), w):.3e})")

    # 1) Gamma^-1 senza taper:  dReInv = -(KK[raw] - ReD_tabella)
    sel = [i for i in range(q.size) if abs(q[i] - qff) <= 0.35]
    dre = {}
    for i in sel:
        dre[i] = -(kk_re_pv_linear(w, raw[i]) - re_tab[i])
    c0 = dre[iq0][w0]
    print(f"\nReGamma^-1 senza taper meno pipeline, tolta la costante {c0 / d:+.2f} delta in (qff,0):")
    ws = [-1.0, -0.5, -0.2, -0.05, 0.0, 0.05, 0.2, 0.5, 1.0]
    print("   Q     " + " ".join(f"{x:+7.2f}" for x in ws) + "   (w; valori in delta)")
    for i in sel:
        vals = [(np.interp(x, w, dre[i]) - c0) / d for x in ws]
        print(f"  {q[i]:.3f} " + " ".join(f"{v:+7.2f}" for v in vals))
    # pendenza in w a (qff,0): pesa il residuo del polo di coppia
    re_inv = np.asarray(z["ReInvGamma"], float)
    m = np.abs(w) < 0.05
    s_pipe = np.polyfit(w[m], re_inv[iq0, m], 1)[0]
    s_diff = np.polyfit(w[m], dre[iq0][m], 1)[0]
    print(f"\ndReGamma^-1/dw a (qff,0): pipeline {s_pipe:+.4e}, variazione senza taper "
          f"{s_diff:+.3e} ({s_diff / s_pipe:+.2%})")
    # curvatura in Q attorno a qff (decide la larghezza in Q della regione critica)
    qs = np.array([q[i] for i in sel])
    base = np.array([re_inv[i, w0] for i in sel])
    corr = np.array([dre[i][w0] - c0 for i in sel])
    k_pipe = np.polyfit(qs - qff, base, 2)[0]
    k_diff = np.polyfit(qs - qff, corr, 2)[0]
    print(f"curvatura d2ReGamma^-1/dQ2 /2 a qff: pipeline {k_pipe:+.4e}, variazione senza taper "
          f"{k_diff:+.3e} ({k_diff / k_pipe:+.2%})")

    # 2) guardia di segno (e floor gia' dentro ImInvGamma salvato? no: floor e' in density)
    imr = np.asarray(z["ImInvGammaRaw"], float)
    im = np.asarray(z["ImInvGamma"], float)
    crit = (np.abs(q - qff) <= 0.35)[:, None] & win[None, :]
    dd = np.abs(im - imr)[crit]
    print(f"\nguardia di segno nella regione critica (|Q-qff|<=0.35, |w|<1): celle toccate "
          f"{int((dd > 0).sum())}/{int(crit.sum())}, max |dImGamma^-1| = {dd.max() / d:.2f} delta")
    return 0


if __name__ == "__main__":
    sys.exit(main())
