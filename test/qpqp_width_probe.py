#!/usr/bin/env python3
"""Quanto del massimo oltre qff viene dalle larghezze QP vicino a kF (regolatori)?  Ricostruzione a posteriori.

Stessa idea di test/qpqp_reverse_probe.py: Gamma^-1_nuovo = Gamma^-1_snap + Delta, dove Delta e' la
differenza fra il pezzo coerente QPxQP con larghezze ridotte e con larghezze originali, calcolate con la
STESSA quadratura fine (quindi Delta misura solo l'effetto delle larghezze):
    Gamma_nuovo(k) = max(Gamma(k) - cut * Gamma_sigma(kF), gamma_min)
(toglie la parte costante, cioe' pavimento di ImSigma + eta, e lascia la crescita non-FL lontano da kF).
Quadratura: gamba up ristretta a kF_up +- shell con nk_shell nodi (la fisica a piccolo Omega), nquad nodi
angolari, Omega fino a wmax (nodi della snap).  Delta g = -4 pi DeltaReGamma^-1(qff, 0), un passo.

Uso (dalla radice di SLIM):  python3 test/qpqp_width_probe.py [RUN] [--cut 0.75] [--workers 8]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import fflo.analytic_bubble as ab  # noqa: E402
from fflo.impi_cv import _canonical_qp_fields, _kf  # noqa: E402
from fflo.qp_cube import qp_model_from_cube  # noqa: E402

DELTA = 1e-3
INK, MUTED, GRID, SURF = "#1f1f1e", "#6b6b68", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False, "font.size": 8.5,
    "legend.frameon": False, "lines.linewidth": 1.5,
})
_W: dict = {}


def _shell_grid(kmax, nk, features, half=0.25, n_local=240):
    """Griglia radiale della gamba up: solo il guscio kF_up +- shell, molto fitto."""
    return np.linspace(_W["kfu"] - _W["shell"], _W["kfu"] + _W["shell"], _W["nk_shell"])


def _row(task):
    j, qv = task
    g = _W
    ab._feature_grid = _shell_grid
    kw = dict(nk=96, n_kquad=g["nquad"], kquad_chunk_size=8, kmax=4.0, kf_features=g["feats"])
    t0 = time.time()
    i_old = ab.coherent_impi_kjac(qv, g["om"], g["ku"], g["eu"], g["zu"], g["gu"], g["kd"], g["ed"], g["zd"], g["gd"],
                                  gamma_floor=1.0e-3, **kw)
    i_new = ab.coherent_impi_kjac(qv, g["om"], g["ku"], g["eu"], g["zu"], g["gu2"], g["kd"], g["ed"], g["zd"], g["gd2"],
                                  gamma_floor=g["gmin"], **kw)
    d_im_ginv = -(i_new - i_old)
    return j, float(np.trapezoid(d_im_ginv / g["om"], g["om"]) / np.pi), time.time() - t0


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", default="P0p50_prod")
    ap.add_argument("--base", default=os.path.join(HERE, "out", "cluster"))
    ap.add_argument("--cut", type=float, default=0.75)
    ap.add_argument("--gamma-min", type=float, default=2.5e-4)
    ap.add_argument("--nquad", type=int, default=960)
    ap.add_argument("--nk-shell", type=int, default=2401)
    ap.add_argument("--shell", type=float, default=0.06)
    ap.add_argument("--wmax", type=float, default=0.12)
    ap.add_argument("--qs", default="0.971,0.989,1.000,1.007,1.025,1.045,1.067")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    snap = sorted(glob.glob(os.path.join(a.base, a.run, "snap", "iter*.npz")))[-1]
    up = sorted(glob.glob(os.path.join(a.base, a.run, "iter*", "next_cubes", "A_komega_spinup_iter*.npz")))[-1]
    dn = up.replace("spinup", "spindown")
    z = np.load(snap)
    q, w, qff = np.asarray(z["q"], float), np.asarray(z["omega"], float), float(z["qff"])
    re_old = np.asarray(z["ReInvGamma"], float)[:, int(np.argmin(np.abs(w)))]
    ku, eu, zu, gu, mu_up = _canonical_qp_fields(up, qp_model_from_cube(up))
    kd, ed, zd, gd, mu_dn = _canonical_qp_fields(dn, qp_model_from_cube(dn))
    feats = [_kf(ku, eu, mu_up), _kf(kd, ed, mu_dn)]
    gku, gkd = float(np.interp(feats[0], ku, gu)), float(np.interp(feats[1], kd, gd))
    gu2 = np.maximum(gu - a.cut * gku, a.gamma_min)
    gd2 = np.maximum(gd - a.cut * gkd, a.gamma_min)
    print(f"{a.run}: Gamma(kF) up {gku:.2e} -> {max(gku * (1 - a.cut), a.gamma_min):.2e}, "
          f"dn {gkd:.2e} -> {max(gkd * (1 - a.cut), a.gamma_min):.2e}; quadratura {a.nquad} angoli, "
          f"{a.nk_shell} nodi in kF_up +- {a.shell}, |Omega| <= {a.wmax}", flush=True)
    wu = np.unique(w)
    om = wu[(np.abs(wu) <= a.wmax) & (np.abs(wu) > 0)]
    targets = [float(x) for x in a.qs.split(",")]
    rows = np.array(sorted({int(np.argmin(np.abs(q / qff - t))) for t in targets}))
    _W.update(om=om, ku=ku, eu=eu, zu=zu, gu=gu, kd=kd, ed=ed, zd=zd, gd=gd, gu2=gu2, gd2=gd2, feats=feats,
              gmin=a.gamma_min, nquad=a.nquad, kfu=feats[0], shell=a.shell, nk_shell=a.nk_shell)
    dre = np.zeros(rows.size)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for j, v, dt in pool.map(_row, [(j, float(q[iq])) for j, iq in enumerate(rows)]):
            dre[j] = v
            print(f"  Q/qff = {q[rows[j]] / qff:.3f}: DeltaReGamma^-1(Q,0) = {v / DELTA:+.3f} delta   [{dt:.0f} s]", flush=True)
    re_new = re_old[rows] + dre
    iqf = int(np.argmin(np.abs(q[rows] - qff)))
    for lab, r in (("snap", re_old[rows]), (f"larghezze ridotte (cut {a.cut})", re_new)):
        i = int(np.argmax(r))
        print(f"{lab:30s}: Q*/qff = {q[rows][i] / qff:.3f}, D = {(r[i] - r[iqf]) / DELTA:+.2f} delta")
    print(f"DeltaReGamma^-1(qff) = {dre[iqf] / DELTA:+.2f} delta  ->  Delta g (un passo) = {-4 * np.pi * dre[iqf]:+.4f}")
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.8), layout="constrained")
    ax.plot(q[rows] / qff, (re_old[rows] - re_old[rows][iqf]) / DELTA, "o-", color="#2a78d6", label="snap")
    ax.plot(q[rows] / qff, (re_new - re_new[iqf]) / DELTA, "o-", color="#eb6834", label=f"larghezze a kF ridotte (cut {a.cut})")
    ax.axvline(1.0, color=MUTED, lw=0.8)
    ax.set(xlabel="Q / qff", ylabel="ReΓ⁻¹(Q, 0) − ReΓ⁻¹(qff, 0)  [δ]", title=f"{a.run}: effetto delle larghezze QP vicino a kF")
    ax.legend(fontsize=7.5)
    out = os.path.join(HERE, "out", "cluster", "plots", f"qpqp_width_{a.run}.png")
    fig.savefig(out, dpi=115)
    print("scritto", out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
