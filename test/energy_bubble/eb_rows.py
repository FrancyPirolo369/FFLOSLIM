#!/usr/bin/env python3
"""Righe ReGamma^-1(Q, 0) con la bolla del prototipo (eb_proto) al posto di quella di produzione.  A un passo.

Per ogni riga Q (vicino a 0 e a qff) e una griglia di Omega in [-12, 12]:
    ImPi_vecchio = residuo a griglia (impi_table: reticolo dw 1e-3, 31 nodi in p, phipanel 10 + 7)
                   + parte coerente QPxQP (fflo.impi_cv, nk 96, 48 nodi angolari)
                   = esattamente la ricetta con cui la run ha fatto la snap
    ImPi_nuovo   = eb_proto.impi (prodotto completo, quadrature adattate)
    DeltaReGamma^-1(Q, 0) = (1/pi) int dOmega [-(ImPi_nuovo - ImPi_vecchio)] / Omega
e le righe nuove sono quelle della snap + Delta.  Ogni riga e' salvata appena finita (ripartenza sicura).
La produzione viene solo letta.

Uso (dalla radice di SLIM):  python3 test/energy_bubble/eb_rows.py [RUN] [--workers 8]
"""
from __future__ import annotations

import argparse
import glob
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "test"))
sys.path.insert(0, os.path.join(HERE, "test", "energy_bubble"))
import eb_proto as E  # noqa: E402

ROWS = [0.0, 0.014, 0.028, 0.057, 0.107, 0.5, 0.95, 1.0, 1.015, 1.041, 1.081]
DELTA = 1e-3
_F = {}


def omega_grid():
    pos = np.geomspace(5e-4, 12.0, 50)
    return np.concatenate((-pos[::-1], pos))


def _new_point(task):
    qv, om = task
    return E.impi(qv, om, _F["up"], _F["dn"], n_pk=16, n_reg=6, n_th_reg=6, n_eps=4)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", default="P0p75_gmax")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", default="", help="indici delle righe da calcolare (es. 6,7,8); le altre se gia' salvate")
    a = ap.parse_args(argv)
    base = os.path.join(HERE, "out", "cluster", a.run)
    up = sorted(glob.glob(os.path.join(base, "iter*", "next_cubes", "A_komega_spinup_iter*.npz")))[-1]
    dn = up.replace("spinup", "spindown")
    it = int(os.path.basename(up).split("iter")[-1][:3])
    z = np.load(os.path.join(base, "snap", f"iter{it:03d}.npz"))
    q, w, qff = np.asarray(z["q"], float), np.asarray(z["omega"], float), float(z["qff"])
    re_snap = np.asarray(z["ReInvGamma"], float)[:, int(np.argmin(np.abs(w)))]
    rows = np.array(sorted({int(np.argmin(np.abs(q / qff - t))) for t in ROWS}))
    rq = q[rows]
    om = omega_grid()
    work = os.path.join(HERE, "out", "energy_bubble", a.run)
    os.makedirs(work, exist_ok=True)
    print(f"{a.run} it{it}: righe Q/qff {np.round(rq / qff, 3)}, {om.size} Omega", flush=True)

    # --- vecchio: residuo + coerente, la ricetta della snap
    old_path = os.path.join(work, "impi_old.npz")
    if os.path.exists(old_path):
        old = np.load(old_path)["impi"]
    else:
        import impi_union_probe as U
        from fflo.qp_cube import build_qp_cube, qp_model_from_cube
        from fflo.impi_cv import _cv_init, _cv_work, _canonical_qp_fields, _kf
        a0u, a0d = os.path.join(work, "A0_up.npz"), os.path.join(work, "A0_dn.npz")
        mu_, md_ = qp_model_from_cube(up), qp_model_from_cube(dn)
        if not os.path.exists(a0u):
            build_qp_cube(up, a0u, model=mu_)
            build_qp_cube(dn, a0d, model=md_)
        t0 = time.time()
        U.setup(up, dn, a0u, a0d, n_p=31, dw=1e-3, n_linear=40)
        res = U.run([float(v) for v in rq], om, "lattice", a.workers)
        ku, eu, zu, gu, mu_up = _canonical_qp_fields(up, mu_)
        kd, ed, zd, gd, mu_dn = _canonical_qp_fields(dn, md_)
        _cv_init(om, ku, eu, zu, gu, kd, ed, zd, gd, [_kf(ku, eu, mu_up), _kf(kd, ed, mu_dn)],
                 dict(angle_mode="kjac", nk=96, nphi=64, n_kquad=48, kquad_chunk_size=8, kmax=4.0, gamma_floor=1.0e-3))
        coh = np.array([_cv_work(float(v)) for v in rq])
        old = res + coh
        np.savez(old_path, q=rq, omega=om, impi=old, residual=res, coherent=coh)
        print(f"  vecchio fatto in {time.time() - t0:.0f} s", flush=True)

    # --- nuovo: prototipo, una riga alla volta
    _F["up"], _F["dn"] = E.Field.from_cube(up, "up"), E.Field.from_cube(dn, "down")
    new = np.full_like(old, np.nan)
    only = {int(x) for x in a.only.split(",")} if a.only else None
    for j, qv in enumerate(rq):
        path = os.path.join(work, f"impi_new_row{j:02d}.npz")
        if os.path.exists(path):
            new[j] = np.load(path)["impi"]
            continue
        if only is not None and j not in only:
            continue
        t0 = time.time()
        with mp.get_context("fork").Pool(a.workers) as pool:
            new[j] = np.array(pool.map(_new_point, [(float(qv), float(o)) for o in om], chunksize=2))
        np.savez(path, q=qv, omega=om, impi=new[j])
        print(f"  riga Q/qff = {qv / qff:.3f}: {time.time() - t0:.0f} s", flush=True)

    dre = np.trapezoid(-(new - old) / om[None, :], om, axis=1) / np.pi     # nan per le righe non calcolate
    r_new = re_snap[rows] + dre
    iq = int(np.argmin(np.abs(rq - qff)))
    i0 = int(np.argmin(np.abs(rq)))
    print("\nQ/qff        " + " ".join(f"{x / qff:7.3f}" for x in rq))
    print("snap   [d]   " + " ".join(f"{(x - re_snap[rows][iq]) / DELTA:+7.2f}" for x in re_snap[rows]))
    print("nuovo  [d]   " + " ".join(f"{(x - r_new[iq]) / DELTA:+7.2f}" for x in r_new))
    g = lambda r: -4 * math.pi * r - 0.5 * math.log(2)
    print(f"ReG^-1(qff) nuovo - snap = {dre[iq] / DELTA:+.2f} delta;  g_c (pin a qff) snap {g(re_snap[rows][iq]):+.4f} -> nuovo {g(r_new[iq]):+.4f}"
          f";  gara ReG(0) - ReG(qff): snap {(re_snap[rows][i0] - re_snap[rows][iq]) / DELTA:+.2f}, nuovo {(r_new[i0] - r_new[iq]) / DELTA:+.2f} delta")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8), layout="constrained")
    for ax, m, t in ((axs[0], rq / qff < 0.6, "vicino a Q = 0"), (axs[1], rq / qff > 0.6, "vicino a qff")):
        ax.plot(rq[m] / qff, (re_snap[rows][m] - re_snap[rows][iq]) / DELTA, "o-", color="#eb6834", label="produzione (snap)")
        ax.plot(rq[m] / qff, (r_new[m] - r_new[iq]) / DELTA, "o-", color="#2a78d6", label="integrazione nuova")
        ax.axhline(0, color="#9a9a96", lw=0.8)
        ax.set(xlabel="Q / qff", ylabel="ReΓ⁻¹(Q,0) − ReΓ⁻¹(qff,0)  [δ]", title=f"{a.run}: {t}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axs[1].axvline(1.0, color="#9a9a96", lw=0.8)
    for j, c in ((i0, "#1baf7a"), (iq, "#4a3aa7")):
        mm = np.abs(om) < 2
        axs[2].plot(om[mm], -old[j][mm], "--", color=c, lw=1.2, label=f"Q = {rq[j] / qff:.2f} qff, produzione")
        axs[2].plot(om[mm], -new[j][mm], "-", color=c, lw=1.4, label=f"Q = {rq[j] / qff:.2f} qff, nuova")
    axs[2].set(xlabel="Ω", ylabel="−ImΠ(Q, Ω)", title="la bolla: prima e dopo")
    axs[2].legend(fontsize=7)
    axs[2].grid(alpha=0.3)
    dst = os.path.join(HERE, "out", "cluster", "plots", f"eb_rows_{a.run}.png")
    fig.savefig(dst, dpi=110)
    print("scritto", dst)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
