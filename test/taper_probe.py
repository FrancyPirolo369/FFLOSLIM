#!/usr/bin/env python3
"""Quanto pesa il taper del residuo (pair_gamma) su ReGamma^-1(Q,0), g_c, canale e contact.

Dalla stessa tabella ImPi (pairbuild --skip-pair) rifa pair_gamma con piu' varianti:
  prod        qkin_cosine stop 20 (produzione)
  stop60      qkin_cosine stop 60
  notaper     nessun taper
  fix_notaper test/fix_truncation.py --mu-mode physical, poi nessun taper
  fix_taper   troncamento corretto + taper di produzione
e per ognuna: massimo di ReGamma^-1(Q,0) in Q <= 2 qff (Q*, g), margini Q=0 e qff, contact
della tabella pinnata al massimo (delta 1e-3, floor exact_zero).
Uso (dalla radice di SLIM):
  python3 test/taper_probe.py IMPI_TABLE.npz [--variants prod,notaper,...] [--out DIR]
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
VARIANTS = {
    "prod": (False, "qkin_cosine", 20.0),
    "stop60": (False, "qkin_cosine", 60.0),
    "notaper": (False, "none", 20.0),
    "fix_notaper": (True, "none", 20.0),
    "fix_taper": (True, "qkin_cosine", 20.0),
}


def pair_gamma(impi, out_dir, mode, stop, log):
    cmd = [sys.executable, "-m", "fflo.pair_gamma", "--no-plots", "--impi-table-path", impi,
           "--kk-pad", "0", "--kk-oversample", "1", "--kk-tail", "zero", "--kk-tail-alpha", "nan",
           "--kk-max-n", "0", "--eta-gamma", "0.002", "--im-inv-sign-guard", "on",
           "--reference-mode", "proxy_residual", "--qp-residual-mode", "cross_inc",
           "--qpqp-source", "auto", "--qpqp-xi-cutoff", "0.015", "--qpqp-n-p", "5000",
           "--qpqp-n-phi", "24", "--residual-omega-taper-start", "0",
           "--residual-omega-taper-stop", f"{stop:g}", "--residual-omega-taper-mode", mode,
           "--out-dir", out_dir]
    with open(log, "w") as fh:
        rc = subprocess.run(cmd, cwd=HERE, stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc:
        raise SystemExit(f"pair_gamma fallito ({log})")
    return os.path.join(out_dir, "pair_gamma_table.npz")


def analyse(path, qff):
    from fflo.pair_shifted import apply_shift_to_pair_table
    from fflo.density import pair_contact_from_shifted_table
    z = np.load(path)
    t = {k: np.asarray(z[k], float) for k in ("q", "omega", "ReInvGamma", "ImInvGamma")}
    q, w = t["q"], t["omega"]
    iw = int(np.argmin(np.abs(w)))
    re0 = t["ReInvGamma"][:, iw]
    win = q <= 2.0 * qff
    iq = int(np.nanargmax(np.where(win, re0, -np.inf)))
    s = apply_shift_to_pair_table(t, float(q[iq]), subcritical_delta=1e-3, eta_floor_mode="exact_zero")
    C = pair_contact_from_shifted_table(s["q"], s["omega"], s["A_pair"])[1]
    i0 = int(np.argmin(np.abs(q))); iff = int(np.argmin(np.abs(q - qff)))
    return dict(q=q, re0=re0, qstar=q[iq] / qff, g=-4 * math.pi * re0[iq] - 0.5 * math.log(2),
                m0=(re0[i0] - re0[iq]) / 1e-3, mff=(re0[iff] - re0[iq]) / 1e-3, C=C)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("impi")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    out = a.out or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(a.impi))), "taper_probe")
    os.makedirs(out, exist_ok=True)
    with np.load(a.impi) as z:
        qff = float(np.sqrt(float(z["mu_up"])) - np.sqrt(float(z["mu_dn"])))
    fixed = os.path.join(out, "impi_table_truncfix.npz")
    res = {}
    for name in a.variants.split(","):
        fix, mode, stop = VARIANTS[name]
        impi = a.impi
        if fix:
            if not os.path.exists(fixed):
                subprocess.run([sys.executable, os.path.join(HERE, "test", "fix_truncation.py"), "--mu-mode",
                                "physical", a.impi, fixed], cwd=HERE, check=True,
                               stdout=open(os.path.join(out, "truncfix.log"), "w"), stderr=subprocess.STDOUT)
            impi = fixed
        tab = pair_gamma(impi, os.path.join(out, name), mode, stop, os.path.join(out, f"{name}.log"))
        res[name] = analyse(tab, qff)
        r = res[name]
        print(f"{name:12s} Q* = {r['qstar']:.3f} qff   g = {r['g']:+.4f}   Q=0 - max {r['m0']:+6.2f} d   "
              f"qff - max {r['mff']:+6.2f} d   C = {r['C']:.3f}", flush=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for name, r in res.items():
        m = r["q"] <= 2.0 * qff
        ax.plot(r["q"][m] / qff, (r["re0"][m] - r["re0"][m].max()) / 1e-3, "-o", ms=3, lw=1, label=name)
    ax.set(xlabel="Q / qff", ylabel="ReΓ⁻¹(Q,0) − max  [δ]", title=os.path.basename(os.path.dirname(os.path.abspath(a.impi))))
    ax.legend(fontsize=8, frameon=False)
    fig.savefig(os.path.join(out, "taper_probe.png"), dpi=120)
    print("scritto", os.path.join(out, "taper_probe.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
