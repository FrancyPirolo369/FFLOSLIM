#!/usr/bin/env python3
"""Lettura delle run a g fissato (test/submit_gscan.sh): g_c per P dove h(g) = g_c(stato) - g = 0.

Per ogni run P0pXX_gs<g>, dall'ultima snapshot:
    g          = -4 pi shift_used - ln2/2            (lo shift fissato)
    g_c(stato) = -4 pi max[ReG^-1(qff,0), ReG^-1(Q0+,0)] - ln2/2   (due candidati; Q0+ = la piu' piccola Q > 0)
    h          = g_c(stato) - g                       > 0: sotto la criticita'
    canale     = quello dei due candidati che e' piu' alto
e la deriva di h sulle ultime 3 iterazioni (convergenza).  Per ogni P, con 2+ punti: g_c dall'interpolazione
lineare di h(g) (o dall'estrapolazione, segnalata).

Uso (dalla radice di SLIM):  python3 test/gscan_report.py [--base out/cluster]
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G0 = -0.5 * math.log(2.0)


def state(path):
    z = np.load(path)
    q, w = np.asarray(z["q"], float), np.asarray(z["omega"], float)
    re0 = np.asarray(z["ReInvGamma"], float)[:, int(np.argmin(np.abs(w)))]
    qff = float(z["qff"])
    iq = int(np.argmin(np.abs(q - qff)))
    nz = np.flatnonzero(q > 1e-9)
    i0 = int(nz[np.argmin(q[nz])])
    best = max(re0[iq], re0[i0])
    g = -4 * math.pi * float(z["shift_used"]) + G0
    gc = -4 * math.pi * best + G0
    return g, gc, ("qff" if re0[iq] >= re0[i0] else "Q~0"), float(z["pair_contact"])


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=os.path.join(HERE, "out", "cluster"))
    a = ap.parse_args(argv)
    by_p = {}
    for d in sorted(glob.glob(os.path.join(a.base, "P0p*_gs*"))):
        snaps = sorted(glob.glob(os.path.join(d, "snap", "iter*.npz")))
        if not snaps:
            continue
        p = float("0." + re.match(r"P0p(\d+)_", os.path.basename(d)).group(1))
        hs = [state(s) for s in snaps[-3:]]
        g, gc, ch, c = hs[-1]
        drift = (hs[-1][1] - hs[0][1]) / max(1, len(hs) - 1)
        by_p.setdefault(p, []).append((g, gc - g, ch, c, drift, len(snaps), os.path.basename(d)))
    if not by_p:
        print("nessuna run P0p*_gs* con snapshot")
        return 1
    for p in sorted(by_p):
        rows = sorted(by_p[p])
        print(f"P = {p:.3f}")
        for g, h, ch, c, dr, n, name in rows:
            print(f"   {name:22s} it{n:3d}  g = {g:+.4f}  h = g_c(stato) - g = {h:+.4f} (deriva {dr:+.4f}/it)  "
                  f"canale {ch:3s}  C = {c:.3f}")
        if len(rows) >= 2:
            gs = np.array([r[0] for r in rows])
            hs = np.array([r[1] for r in rows])
            k = np.polyfit(gs, hs, 1)
            gc = -k[1] / k[0]
            inside = gs.min() <= gc <= gs.max()
            print(f"   -> g_c = {gc:+.4f}  ({'interpolato' if inside else 'ESTRAPOLATO'}; pendenza dh/dg = {k[0]:+.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
