#!/usr/bin/env python3
"""Stato delle run dai loop.log: contact, deriva di g_c, gap del minoritario, Q selezionato.

Criterio di convergenza = plateau del contact (la densita' converge troppo presto per
discriminare).  g_inf e' una stima geometrica: se le ultime due derive di g stanno in
rapporto r < 0.97, g_inf = g + dg * r / (1 - r); altrimenti '-' (la deriva non cala).

Uso (da test/, sul cluster o in locale):
  python3 status_runs.py [--base ../out] [RUN_GLOB ...]      default: P0p*
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import re


def parse(loop_log):
    it, qff = {}, None
    for line in open(loop_log):
        m = re.search(r"qff=([\d.]+)", line)
        if m and qff is None:
            qff = float(m[1])
        m = re.search(r"iter (\d+) PAIR: contact=([\d.]+).*shift=([-+\d.eE]+)(?:.*Q=([\d.]+))?", line)
        if m:
            it.setdefault(int(m[1]), {}).update(
                C=float(m[2]), g=-4 * math.pi * float(m[3]) - 0.5 * math.log(2),
                Q=float(m[4]) if m[4] else None)
        m = re.search(r"iter (\d+) GAP: up=([-+\d.eE]+)% down=([-+\d.eE]+)%", line)
        if m:
            it.setdefault(int(m[1]), {}).update(gd=float(m[3]))
    return {k: v for k, v in it.items() if "C" in v and "gd" in v}, qff


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", default=["P0p*"])
    ap.add_argument("--base", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out"))
    a = ap.parse_args()
    dirs = sorted({d for pat in a.runs for d in glob.glob(os.path.join(a.base, pat))
                   if os.path.exists(os.path.join(d, "loop.log"))})
    print(f"{'run':18s} it    C    C/prev (ultime 3)        g_c    dg/it (ultime 3)        g_inf   gap_dn  Q/qff")
    for d in dirs:
        it, qff = parse(os.path.join(d, "loop.log"))
        ks = sorted(it)
        name = os.path.basename(d)
        if len(ks) < 2:
            print(f"{name:18s} {ks[-1] if ks else 0:2d}  (troppo poche iterazioni)")
            continue
        last = ks[-4:]
        rat = [it[last[i]]["C"] / it[last[i - 1]]["C"] for i in range(1, len(last))]
        dg = [it[last[i]]["g"] - it[last[i - 1]]["g"] for i in range(1, len(last))]
        r = it[ks[-1]]
        ginf = "   -  "
        if len(dg) >= 2 and dg[-2] != 0.0:
            q = dg[-1] / dg[-2]
            if 0.0 < q < 0.97:
                ginf = f"{r['g'] + dg[-1] * q / (1.0 - q):+.3f}"
        qrel = f"{r['Q'] / qff:.3f}" if (r.get("Q") is not None and qff) else "qff"
        print(f"{name:18s} {ks[-1]:2d}  {r['C']:.3f}  " + " ".join(f"{x:.4f}" for x in rat).ljust(20)
              + f"   {r['g']:+.3f}  " + " ".join(f"{x:+.4f}" for x in dg).ljust(22)
              + f"  {ginf}  {r['gd']:+6.2f}%  {qrel}")


if __name__ == "__main__":
    main()
