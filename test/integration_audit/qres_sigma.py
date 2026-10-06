#!/usr/bin/env python3
"""Risoluzione in Q della tabella di coppia, effetto su Sigma: righe di produzione contro righe raddoppiate.

Tabella da una sola corsa di pairbuild (bubble_audit.py --qset dense: tutte le righe Q di produzione piu'
i punti medi fino a 4 qff, stessa griglia Omega per tutte), poi due tabelle:
    prodQ   solo le righe che coincidono con la griglia Q di produzione (snap della stessa iterazione)
    denseQ  tutte le righe
stesso shift di Thouless (dalla riga qff, delta = 1e-3), stessa Sigma di produzione (sigma_audit.py:
cube d'ingresso della iterazione, righe k e nodi omega di produzione).  La differenza denseQ - prodQ e'
l'errore di risoluzione in Q della produzione su ImSigma.

Uso (dalla radice di SLIM, col tetto di memoria):
    systemd-run --user --scope -p MemoryMax=5G python3 test/integration_audit/qres_sigma.py compute
    python3 test/integration_audit/qres_sigma.py report
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sigma_audit as S  # noqa: E402  (imposta le SIGMA_PN_* di produzione)
import bubble_audit as B  # noqa: E402

import argparse  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from fflo.density import memory_safe_im_sigma  # noqa: E402
from fflo.pair_shifted import apply_shift_to_pair_table  # noqa: E402


def out_path(run, it):
    return os.path.join(S.HERE, "out", "integration_audit", f"qres_sigma_{run}_it{it}.npz")


def tables(run, it, base):
    q, w, re, im = B.load_pair(os.path.join(B.work_dir(run, it, "dense"), "base", "pair", "pair_gamma_table.npz"))
    z = np.load(os.path.join(base, run, "snap", f"iter{it:03d}.npz"))
    zq = np.asarray(z["q"], float)
    prod = np.array([np.min(np.abs(zq - v)) < 1e-9 for v in q])
    q_sel = float(z["q_selected"])
    out = {}
    for name, m in (("prodQ", prod), ("denseQ", np.ones(q.size, bool))):
        t = {"q": q[m], "omega": w, "ReInvGamma": re[m], "ImInvGamma": im[m]}
        sh = apply_shift_to_pair_table(t, q_sel, subcritical_delta=S.DELTA, eta_floor_mode="exact_zero")
        out[name] = (np.asarray(sh["q"], float), np.asarray(sh["omega"], float), np.asarray(sh["A_pair"], float))
    return out, int(prod.sum()), q.size


def compute(a):
    ctx = S.setup(a.run, a.it, a.base)
    tabs, n_prod, n_all = tables(a.run, a.it, a.base)
    print(f"tabella: {n_prod} righe di produzione, {n_all} in tutto", flush=True)
    path = out_path(a.run, a.it)
    res = dict(np.load(path, allow_pickle=True)) if os.path.exists(path) else {}
    for name, (pq, pw, pa) in tabs.items():
        for spin, tag, mukey in S.SPINS:
            if f"im_{tag}_{name}" in res:
                continue
            kf, idx, rows = S.rows_for(ctx, tag, mukey)
            w = np.asarray(ctx["s"][f"ck_{tag}_omega"], float).ravel()
            t0 = time.time()
            im = memory_safe_im_sigma(
                spin=spin, cube=ctx["seeds"]["down" if spin == "up" else "up"],
                pair_q=pq, pair_omega=pw, pair_a=pa, k_target=rows, omega_target=w,
                physics=ctx["phys"], qff=ctx["q_sel"], sigma_nk=rows.size,
                sigma_k_feature_fraction=0.0, n_theta=32, omega_chunk=1000,
                workers=a.workers if name == "prodQ" else max(1, a.workers - 1),
                checkpoint=Path(os.path.dirname(path)) / f"ck_qres_{name}_{tag}.npz")
            res.update({f"im_{tag}_{name}": im, f"rows_{tag}": rows, f"w_{tag}": w, f"kf_{tag}": kf})
            np.savez_compressed(path, **res)
            print(f"[{name}] {tag}: {time.time() - t0:.0f} s", flush=True)
    return 0


def report(a):
    res = dict(np.load(out_path(a.run, a.it), allow_pickle=True))
    hdr = "  ".join(f"{lo:g}-{hi:g}" if hi < 1e8 else f">{lo:g}" for lo, hi in S.BANDS)
    print("ImSigma(denseQ) - ImSigma(prodQ), relativo con segno (+ = |ImSigma| cala con piu' righe Q)")
    for _, tag, _ in S.SPINS:
        if f"im_{tag}_denseQ" not in res:
            continue
        rows, kf, w = res[f"rows_{tag}"], float(res[f"kf_{tag}"]), res[f"w_{tag}"]
        print(f"=== Sigma_{tag} (kF = {kf:.4f})   bande |omega|: {hdr}")
        for i, kv in enumerate(rows):
            e = S.band_err(w, res[f"im_{tag}_prodQ"][i], res[f"im_{tag}_denseQ"][i], signed=True)
            print(f"   k = {kv:.4f} ({kv / kf:.3f} kF)   " + "  ".join(f"{v:+9.2e}" for v in e))
    return 0


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("compute", "report"))
    ap.add_argument("--run", default="P0p50_final")
    ap.add_argument("--it", type=int, default=8)
    ap.add_argument("--base", default=os.path.join(S.HERE, "out", "cluster"))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    return compute(a) if a.mode == "compute" else report(a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
