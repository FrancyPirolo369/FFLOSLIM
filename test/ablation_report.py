#!/usr/bin/env python3
"""Lettura dei test di ricetta di test/submit_ablation.sh.

Per ogni variante P0pXX_abl_<nome>: C, g_c e gap_dn alle iterazioni 1 e 2, e lo scarto dal
controllo T0_ctrl.  Manopole della coppia (T1, R3, R4) si leggono gia' all'iterazione 1; quelle
di Sigma (T2-T5, R1, R2) all'iterazione 2 (la loro iterazione 1 deve coincidere con T0: e' la
misura del rumore).  In fondo il riferimento: ultima iterazione di etaexact e di prod.

Uso (dalla radice di SLIM o da test/):  python3 test/ablation_report.py [--P 0.50] [--base out/cluster]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "test"))
from status_runs import parse  # noqa: E402

DELTA = 1e-3

ORDER = ["T0_ctrl", "T1_tail20", "T2_som41", "T3_stale", "T4_notail", "T5_etaexact",
         "R1_som193", "R2_snk48", "R3_dw", "R4_pnodes", "R5_pnodes4", "R6_lam6", "R7_gold", "R8_qff"]
PAIR_SIDE = {"T1_tail20", "R3_dw", "R4_pnodes", "R5_pnodes4", "R6_lam6", "R7_gold", "R8_qff"}


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--P", default="0.50")
    ap.add_argument("--base", default=os.path.join(HERE, "out", "cluster"))
    a = ap.parse_args(argv)
    tag = "P" + a.P.replace(".", "p")
    rows = {}
    for name in ORDER:
        log = os.path.join(a.base, f"{tag}_abl_{name}", "loop.log")
        if os.path.exists(log):
            rows[name] = parse(log)[0]
    if not rows:
        print(f"nessuna run {tag}_abl_* in {a.base}")
        return 1
    ctrl = rows.get("T0_ctrl", {})
    print(f"{tag}: risposta a un passo (alpha = 1) dallo stato convergito di prod.  d = variante - T0_ctrl")
    print(f"{'variante':13s} {'lato':6s} | {'C it1':>8s} {'g it1':>8s} | {'C it2':>8s} {'g it2':>8s} {'gap_dn2':>8s} |"
          f" {'dC/C it2':>9s} {'dg it2':>8s}")
    for name in ORDER:
        if name not in rows:
            continue
        r = rows[name]
        side = "coppia" if name in PAIR_SIDE else "Sigma"

        def val(i, k):
            return r[i][k] if i in r else float("nan")

        line = (f"{name:13s} {side:6s} | {val(1, 'C'):8.4f} {val(1, 'g'):+8.4f} | {val(2, 'C'):8.4f} "
                f"{val(2, 'g'):+8.4f} {val(2, 'gd'):+7.2f}% |")
        if name != "T0_ctrl" and 2 in r and 2 in ctrl:
            line += f" {100 * (r[2]['C'] / ctrl[2]['C'] - 1):+8.2f}% {r[2]['g'] - ctrl[2]['g']:+8.4f}"
        print(line)
    for fam in ("etaexact", "prod"):
        log = os.path.join(a.base, f"{tag}_{fam}", "loop.log")
        if os.path.exists(log):
            it = parse(log)[0]
            k = max(it)
            print(f"riferimento {fam:9s} it{k:02d}: C = {it[k]['C']:.4f}, g = {it[k]['g']:+.4f}, gap_dn = {it[k]['gd']:+.2f}%")
    # In 2D il massimo di ReGamma^-1(Q, 0) deve stare esattamente a qff = kF_up - kF_dn: il dislivello
    # D = ReGamma^-1(Q*) - ReGamma^-1(qff) [delta] e Q*/qff misurano l'errore numerico (bersaglio 0 e 1).
    print("\ndislivello oltre la tangenza (dalle snap):  D [delta] e Q*/qff per iterazione")
    for name in ["prod"] + ORDER:
        run = f"{tag}_{name}" if name == "prod" else f"{tag}_abl_{name}"
        snaps = sorted(glob.glob(os.path.join(a.base, run, "snap", "iter*.npz")))
        if name == "prod":
            snaps = snaps[-1:]
        cells = []
        for s in snaps:
            z = np.load(s)
            q, w, qff = np.asarray(z["q"], float), np.asarray(z["omega"], float), float(z["qff"])
            re = np.asarray(z["ReInvGamma"], float)[:, int(np.argmin(np.abs(w)))]
            iq = int(np.argmax(np.where(q <= 2 * qff, re, -np.inf)))
            d = (re[iq] - re[int(np.argmin(np.abs(q - qff)))]) / DELTA
            cells.append(f"it{s[-7:-4]} D = {d:4.2f} Q* = {q[iq] / qff:.3f}")
        if cells:
            print(f"  {name:13s} " + " | ".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
