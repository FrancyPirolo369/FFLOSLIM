#!/usr/bin/env python3
"""Convergenza in p della bolla a P = 0.90 (stato t_frz it12 -> it13) contro il conto statico esatto (2026-10-05).

Legge le varianti presenti in out/integration_audit/bubble_P0p90_t_frz_it13_large/ (calcolate in locale o sul
cluster con test/run_bubble.slurm) e out/static_gamma/P0p90_largeQ.npz (test/static_gamma.py, locale).  Stampa:
  - parte su griglia: KK a Omega = 0 col peso del taper HP, relativa a qff (e assoluta a Q = 0, qff);
  - parte coerente analitica: idem;
  - totale HP: scarto dall'esatto relativo a qff, g_c e gara.
Uso (dalla radice di SLIM):  python3 test/integration_audit/compare_pconv.py [VARIANTE ...]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from fflo.pair_gamma import kk_pv_linear_at  # noqa: E402

D = os.path.join(HERE, "out", "integration_audit", "bubble_P0p90_t_frz_it13_large")
DEFAULT = ["f2", "f2p31c192", "f2p62c192", "f2L8ca", "f2L8c384", "f2L8p100c384", "f2L8p140", "f2L8p200c384"]
QFF = np.sqrt(1.9) - np.sqrt(0.1)
DL = 1e-3


def gc(v):
    return -4.0 * np.pi * v - 0.5 * np.log(2.0)


def main(argv):
    names = [v for v in (argv or DEFAULT)
             if os.path.exists(os.path.join(D, v, "pair_hp", "pair_gamma_table.npz"))]
    s = np.load(os.path.join(HERE, "out", "static_gamma", "P0p90_largeQ.npz"))
    sq, ex = s["q"], s["re_inv_exact"]
    isq = int(np.argmin(np.abs(sq - QFF)))
    parts = {}
    for v in names:
        z = np.load(os.path.join(D, v, "impi", "impi_table.npz"))
        p = np.load(os.path.join(D, v, "pair_hp", "pair_gamma_table.npz"))
        q, w, wt = z["q"], z["omega"], p["residual_omega_taper_weight"]
        g = {}
        for j, Q in enumerate(q):
            if Q <= 8.0:
                g[round(float(Q), 4)] = tuple(kk_pv_linear_at(w, z[k][j] * wt[j], np.array([0.0]))[0]
                                              for k in ("ImPiGridResidual", "ImPiQPQP"))
        parts[v] = (g, p, float(z["p_int_max"]))
    print("varianti:", ", ".join(f"{v} (Lambda {parts[v][2]:g})" for v in names))
    for k, lab in ((0, "GRIGLIA A*A - A0*A0"), (1, "COERENTE A0*A0")):
        print(f"\n{lab}: KK a 0 col taper HP, relativa a qff [delta]")
        print("   Q      " + " ".join(f"{v:>13s}" for v in names))
        qs = sorted(parts[names[0]][0])
        for Q in qs:
            row = []
            for v in names:
                g = parts[v][0]
                jq = min(g, key=lambda x: abs(x - QFF))
                row.append((g[Q][k] - g[jq][k]) / DL if Q in g else np.nan)
            print(f"  {Q:6.3f} " + " ".join(f"{x:+13.2f}" for x in row))
        for Q0 in (0.0,):
            print("  assoluto Q=0: " + " ".join(f"{parts[v][0][Q0][k] / DL:+13.2f}" for v in names))
    print("\nTOTALE HP: scarto dall'esatto relativo a qff [delta]")
    print("   Q      " + " ".join(f"{v:>13s}" for v in names))
    for i, Q in enumerate(sq):
        row = []
        for v in names:
            p = parts[v][1]
            tq = p["q"]
            i0 = int(np.argmin(np.abs(p["omega"])))
            j = int(np.argmin(np.abs(tq - Q)))
            jq = int(np.argmin(np.abs(tq - QFF)))
            ok = abs(tq[j] - Q) < 1e-6 * max(1.0, Q)
            row.append(((p["ReInvGamma"][j, i0] - ex[i]) - (p["ReInvGamma"][jq, i0] - ex[isq])) / DL if ok else np.nan)
        print(f"  {Q:6.3f} " + " ".join(f"{x:+13.2f}" for x in row))
    print(f"\nesatto: g = {gc(ex[isq]):+.4f}, gara = {(ex[0] - ex[isq]) / DL:+.2f} delta")
    for v in names:
        p = parts[v][1]
        tq = p["q"]
        i0 = int(np.argmin(np.abs(p["omega"])))
        jq = int(np.argmin(np.abs(tq - QFF)))
        jz = int(np.argmin(np.abs(tq)))
        print(f"  {v:14s}: g = {gc(p['ReInvGamma'][jq, i0]):+.4f}  gara = "
              f"{(p['ReInvGamma'][jz, i0] - p['ReInvGamma'][jq, i0]) / DL:+.2f} delta")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
