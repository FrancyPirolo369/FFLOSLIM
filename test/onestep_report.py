#!/usr/bin/env python3
"""Tabella dei job a un passo (submit.sh L0..L5): eccesso FRESCO per spin.

Uso (dalla radice):  python3 test/onestep_report.py [RUN ...]   (default: out/L*)
Per ogni run legge iter001/density: density_results.npz (densita' con alpha=1,
cioe' dove punta il punto fisso) e density_summary.txt (contact, shift).
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def grab(text, key):
    for line in text.splitlines():
        if line.startswith(key + " "):
            try:
                return float(line.split()[1])
            except (IndexError, ValueError):
                return float("nan")
    return float("nan")


def main(argv):
    runs = argv or sorted(os.path.basename(p) for p in glob.glob(os.path.join(ROOT, "out", "L*")))
    print(f"{'run':16s} {'C':>7s} {'shift':>9s} {'dn - mu/2':>10s} {'up - mu/2':>10s} "
          f"{'dn/up':>6s} {'(dn,up)/C':>14s}")
    for r in runs:
        d = os.path.join(ROOT, "out", r, "iter001", "density")
        try:
            z = np.load(os.path.join(d, "density_results.npz"))
            txt = open(os.path.join(d, "density_summary.txt")).read()
        except OSError:
            print(f"{r:16s} (non ancora completo)")
            continue
        c = grab(txt, "pair_contact")
        sh = grab(txt, "pair_raw_thouless_shift")
        dn = float(z["alpha_1_density_down"]) - 0.175
        up = float(z["alpha_1_density_up"]) - 0.825
        print(f"{r:16s} {c:7.4f} {sh:+9.5f} {dn:+10.5f} {up:+10.5f} {dn / up:6.2f} "
              f"{dn / c:+7.4f},{up / c:+7.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
