#!/usr/bin/env python3
"""Come sarebbe stata ReGamma^-1(Q, 0) con il pezzo coerente QPxQP integrato bene (ricostruzione a posteriori).

Dalla snap di una run: Gamma^-1 della tabella della coppia.  Dalle cube della stessa iterazione: i campi QP
(stessa estrazione di pairbuild: qp_model_from_cube + _canonical_qp_fields, kF di feature da _kf).  Si ricalcola
ImPi_coh(Q, Omega) con la kjac di produzione (48 nodi angolari) e con una kjac convergita (--nquad-fine), e
    Gamma^-1_nuovo = Gamma^-1_snap + DeltaGamma^-1,   DeltaImGamma^-1 = -(ImPi_fine - ImPi_48)
(convenzione della snap: ImGamma^-1 -> +1/8 a Omega grande), con la parte reale dalla KK a Omega = 0:
    DeltaReGamma^-1(Q, 0) = (1/pi) int dOmega DeltaImGamma^-1(Q, Omega) / Omega.
E' un passo solo (Sigma e cube invariate).  Lo shift del pin e' ReGamma^-1(qff, 0) grezzo, quindi
Delta g = -4 pi DeltaReGamma^-1(qff, 0).

Uso (dalla radice di SLIM):  python3 test/qpqp_reverse_probe.py [RUN] [--nquad-fine 480] [--wmax 20]
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
from fflo.analytic_bubble import coherent_impi_kjac  # noqa: E402
from fflo.impi_cv import _canonical_qp_fields, _kf  # noqa: E402
from fflo.qp_cube import qp_model_from_cube  # noqa: E402

DELTA = 1e-3
_W: dict = {}


def _row(task):
    """Una riga Q: DeltaReGamma^-1(Q, 0) fra kjac a 48 nodi e kjac fine (processo separato)."""
    j, qv = task
    g = _W
    kw = dict(nk=96, kquad_chunk_size=8, kmax=4.0, gamma_floor=1.0e-3, kf_features=g["feats"])
    args = (qv, g["om"], g["ku"], g["eu"], g["zu"], g["gu"], g["kd"], g["ed"], g["zd"], g["gd"])
    t0 = time.time()
    i48 = coherent_impi_kjac(*args, n_kquad=48, **kw)
    ifi = coherent_impi_kjac(*args, n_kquad=g["nfine"], **kw)
    d_im_ginv = -(ifi - i48)
    return j, float(np.trapezoid(d_im_ginv / g["om"], g["om"]) / np.pi), time.time() - t0


INK, MUTED, GRID, SURF = "#1f1f1e", "#6b6b68", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False, "font.size": 8.5,
    "legend.frameon": False, "lines.linewidth": 1.5,
})


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", default="P0p50_prod")
    ap.add_argument("--base", default=os.path.join(HERE, "out", "cluster"))
    ap.add_argument("--nquad-fine", type=int, default=480)
    ap.add_argument("--wmax", type=float, default=20.0)
    ap.add_argument("--qlo", type=float, default=0.92)
    ap.add_argument("--qhi", type=float, default=1.10)
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
    print(f"{a.run}: snap {os.path.basename(snap)}, cube {os.path.basename(up)}, kF di feature {feats[0]:.4f}/{feats[1]:.4f}, "
          f"Z(kF) up {np.interp(feats[0], ku, zu):.3f} dn {np.interp(feats[1], kd, zd):.3f}")
    wu, iu = np.unique(w, return_index=True)
    m = (np.abs(wu) <= a.wmax) & (np.abs(wu) > 0)
    om = wu[m]
    rows = np.flatnonzero((q >= a.qlo * qff) & (q <= a.qhi * qff))
    dre = np.zeros(rows.size)
    _W.update(om=om, ku=ku, eu=eu, zu=zu, gu=gu, kd=kd, ed=ed, zd=zd, gd=gd, feats=feats, nfine=a.nquad_fine)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for j, v, dt in pool.map(_row, [(j, float(q[iq])) for j, iq in enumerate(rows)]):
            dre[j] = v
            print(f"  Q/qff = {q[rows[j]] / qff:.3f}: DeltaReGamma^-1(Q,0) = {v / DELTA:+.3f} delta   [{dt:.0f} s]", flush=True)
    re_new = re_old[rows] + dre
    iqf = int(np.argmin(np.abs(q[rows] - qff)))
    for lab, r in (("snap (kjac 48)", re_old[rows]), (f"ricostruita (kjac {a.nquad_fine})", re_new)):
        i = int(np.argmax(r))
        print(f"{lab:26s}: Q*/qff = {q[rows][i] / qff:.3f}, D = {(r[i] - r[iqf]) / DELTA:+.2f} delta")
    print(f"DeltaReGamma^-1(qff) = {dre[iqf] / DELTA:+.2f} delta  ->  Delta g (un passo) = {-4 * np.pi * dre[iqf]:+.4f}")
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.8), layout="constrained")
    ax.plot(q[rows] / qff, (re_old[rows] - re_old[rows][iqf]) / DELTA, "o-", color="#2a78d6", label="snap (kjac 48 nodi)")
    ax.plot(q[rows] / qff, (re_new - re_new[iqf]) / DELTA, "o-", color="#eb6834", label=f"ricostruita (kjac {a.nquad_fine} nodi)")
    ax.axvline(1.0, color=MUTED, lw=0.8)
    ax.set(xlabel="Q / qff", ylabel="ReΓ⁻¹(Q, 0) − ReΓ⁻¹(qff, 0)  [δ]", title=f"{a.run}: pezzo coerente integrato bene")
    ax.legend(fontsize=7.5)
    out = os.path.join(HERE, "out", "cluster", "plots", f"qpqp_reverse_{a.run}.png")
    fig.savefig(out, dpi=115)
    print("scritto", out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
