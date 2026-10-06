#!/usr/bin/env python3
"""Audit degli integrali di Sigma: stessa iterazione della produzione, una manopola alla volta.

Stato: RUN, iterazione IT.  Ingressi identici a quelli di density in quella iterazione:
    cube (fermione interno)  = out/cluster/RUN/iter{IT-1}/next_cubes   (le cube mescolate d'ingresso)
    tabella di coppia        = snap/iter{IT}.npz (ReInvGamma, ImInvGamma) con lo shift di Thouless
                               a q_selected e delta = 1e-3 (apply_shift_to_pair_table, exact_zero)
    righe k, nodi omega      = quelli di produzione (snap/sigma{IT}.npz: ck_*_k_sparse, ck_*_omega)
La variante "base" deve riprodurre ck_*_im_sparse della snap (convalida del banco).

Varianti (una manopola alla volta, il resto come in produzione):
    theta64    n_theta 32 -> 64                    (media angolare della parte liscia del fermione)
    qgl32      SIGMA_PN_Q_GL_N 16 -> 32           (Gauss-Legendre in Q dentro ogni cella della tabella)
    qtab4      tabella in Q raffinata x4 su [0, 3 qff] interpolando Gamma^-1 (Re, Im) e ricalcolando
               A_pair (in produzione A_pair e' interpolata linearmente fra righe distanti ~0.03 qff:
               misura quanto pesa la forma del picco critico fra le righe)
    epscore    SIGMA_PN_EPS_CORE_DW 2e-3 -> 5e-4  (passo della tabella del fermione T(q, eps) vicino a 0)
    ring       SIGMA_PN_EXACT_WINDOW_GAMMA 16 -> 32, SIGMA_PN_RING_NWIN 48 -> 96 (finestra del polo QP)
    tailcell   SIGMA_PN_TAILCELL 0.02 -> 0.005    (suddivisione delle celle larghe in Omega)
    pint16     SIGMA_PN_PINTMAX 12 -> 16          (taglio del momento interno)

Uscita: out/integration_audit/sigma_<RUN>_it<IT>.npz e tabella degli scarti relativi per riga e banda
di omega:  sum|d ImSigma| / sum|ImSigma|  su |w| < 0.02, 0.02-0.2, 0.2-2, 2-20, > 20.

Uso (dalla radice di SLIM):
    python3 test/integration_audit/sigma_audit.py compute --run P0p50_final --it 8 --variants base,theta64
    python3 test/integration_audit/sigma_audit.py report  --run P0p50_final --it 8
"""
from __future__ import annotations

import os

os.environ.setdefault("SIGMA_PN_PINTMAX", "12")
os.environ.setdefault("SIGMA_PN_RING_MODE", "exact_window")
os.environ.setdefault("SIGMA_PN_ANGLE_EXACT_CUT", "1")
os.environ.setdefault("SIGMA_PN_Q_GL_N", "16")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import glob  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from fflo.density import load_seed, memory_safe_im_sigma  # noqa: E402
from fflo.pair_shifted import apply_shift_to_pair_table  # noqa: E402
from fflo.sigma_engine import Physics  # noqa: E402

DELTA = 1.0e-3
SPINS = (("down", "dn", "mu_dn"), ("up", "up", "mu_up"))
ROW_TARGETS = (0.5, 0.97, 1.0, 1.03, 1.3, 2.0)       # in unita' di kF, piu' k = 4 e 7
BANDS = ((0.0, 0.02), (0.02, 0.2), (0.2, 2.0), (2.0, 20.0), (20.0, 1e9))
VARIANTS = {
    "base": ({}, 32, False),
    "theta64": ({}, 64, False),
    "qgl32": ({"SIGMA_PN_Q_GL_N": "32"}, 32, False),
    "qtab4": ({}, 32, True),
    "epscore": ({"SIGMA_PN_EPS_CORE_DW": "5e-4"}, 32, False),
    "ring": ({"SIGMA_PN_EXACT_WINDOW_GAMMA": "32", "SIGMA_PN_RING_NWIN": "96"}, 32, False),
    "tailcell": ({"SIGMA_PN_TAILCELL": "0.005"}, 32, False),
    "pint16": ({"SIGMA_PN_PINTMAX": "16"}, 32, False),
}


# varianti che ingrandiscono gli array per worker (q x nodi Omega del segmento): meno processi
# (2026-10-02: qgl32 con 4 worker ha sforato il tetto di 4 GB)
HEAVY_WORKERS = {"qgl32": 2, "qtab4": 2, "tailcell": 1}


def out_path(run, it):
    d = os.path.join(HERE, "out", "integration_audit")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"sigma_{run}_it{it}.npz")


def setup(run, it, base):
    rd = os.path.join(base, run)
    cdir = os.path.join(rd, f"iter{it - 1:03d}", "next_cubes")
    up = glob.glob(os.path.join(cdir, "A_komega_spinup_iter*.npz"))[0]
    dn = glob.glob(os.path.join(cdir, "A_komega_spindown_iter*.npz"))[0]
    seeds = {"up": load_seed(Path(up)), "down": load_seed(Path(dn))}
    z = np.load(os.path.join(rd, "snap", f"iter{it:03d}.npz"))
    s = np.load(os.path.join(rd, "snap", f"sigma{it:03d}.npz"), allow_pickle=True)
    table = {"q": np.asarray(z["q"], float), "omega": np.asarray(z["omega"], float),
             "ReInvGamma": np.asarray(z["ReInvGamma"], float),
             "ImInvGamma": np.asarray(z["ImInvGamma"], float)}
    q_sel = float(z["q_selected"])
    phys = Physics(eps0=1.0, mu_up=seeds["up"]["mu_up"], mu_down=seeds["up"]["mu_dn"],
                   mass=seeds["up"]["mass"], pmf_scale=0.0)
    return dict(seeds=seeds, table=table, q_sel=q_sel, qff=float(z["qff"]), s=s, phys=phys)


def shifted(table, q_sel, refine):
    t = table
    if refine:
        q = t["q"]
        qmax = 3.0 * q_sel
        fine = [q[0]]
        for a, b in zip(q[:-1], q[1:]):
            n = 4 if b <= qmax + 1e-12 else 1
            fine.extend(np.linspace(a, b, n + 1)[1:])
        qf = np.asarray(fine)
        re = np.array([np.interp(qf, q, t["ReInvGamma"][:, j]) for j in range(t["omega"].size)]).T
        im = np.array([np.interp(qf, q, t["ImInvGamma"][:, j]) for j in range(t["omega"].size)]).T
        t = {"q": qf, "omega": t["omega"], "ReInvGamma": re, "ImInvGamma": im}
    sh = apply_shift_to_pair_table(t, q_sel, subcritical_delta=DELTA, eta_floor_mode="exact_zero")
    return np.asarray(sh["q"], float), np.asarray(sh["omega"], float), np.asarray(sh["A_pair"], float)


def rows_for(ctx, tag, mukey):
    kf = float(np.sqrt(ctx["seeds"]["up"]["mass"] * ctx["seeds"]["up"][mukey]))
    ks = np.asarray(ctx["s"][f"ck_{tag}_k_sparse"], float).ravel()
    want = [r * kf for r in ROW_TARGETS] + [4.0, 7.0]
    idx = sorted({int(np.argmin(np.abs(ks - v))) for v in want})
    return kf, np.asarray(idx), ks[idx]


def compute(a):
    ctx = setup(a.run, a.it, a.base)
    path = out_path(a.run, a.it)
    res = dict(np.load(path, allow_pickle=True)) if os.path.exists(path) else {}
    for name in a.variants.split(","):
        env, n_theta, refine = VARIANTS[name]
        if f"im_dn_{name}" in res and f"im_up_{name}" in res and not a.force:
            print(f"[{name}] gia' fatto, salto")
            continue
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            pq, pw, pa = shifted(ctx["table"], ctx["q_sel"], refine)
            for spin, tag, mukey in SPINS:
                kf, idx, rows = rows_for(ctx, tag, mukey)
                w = np.asarray(ctx["s"][f"ck_{tag}_omega"], float).ravel()
                t0 = time.time()
                ck = Path(os.path.dirname(path)) / f"ck_{a.run}_it{a.it}_{name}_{tag}.npz"
                im = memory_safe_im_sigma(
                    spin=spin, cube=ctx["seeds"]["down" if spin == "up" else "up"],
                    pair_q=pq, pair_omega=pw, pair_a=pa, k_target=rows, omega_target=w,
                    physics=ctx["phys"], qff=ctx["q_sel"], sigma_nk=rows.size,
                    sigma_k_feature_fraction=0.0, n_theta=n_theta, omega_chunk=1000,
                    workers=min(a.workers, HEAVY_WORKERS.get(name, a.workers)), checkpoint=ck)
                res.update({f"im_{tag}_{name}": im, f"rows_{tag}": rows, f"idx_{tag}": idx,
                            f"w_{tag}": w, f"kf_{tag}": kf,
                            f"prod_{tag}": np.asarray(ctx["s"][f"ck_{tag}_im_sparse"], float)[idx]})
                np.savez_compressed(path, **res)
                if ck.exists():
                    ck.unlink()
                print(f"[{name}] {tag}: {rows.size} righe x {w.size} omega in {time.time() - t0:.0f} s",
                      flush=True)
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    return 0


def band_err(w, ref, x, signed=False):
    """scarto relativo per banda: L1, oppure (signed) sum(x - ref) / sum|ref| (> 0: |ImSigma| piu' piccola)"""
    out = []
    for lo, hi in BANDS:
        m = (np.abs(w) >= lo) & (np.abs(w) < hi)
        den = np.sum(np.abs(ref[m]))
        num = np.sum(x[m] - ref[m]) if signed else np.sum(np.abs(x[m] - ref[m]))
        out.append(num / den if den > 0 else np.nan)
    return out


def report(a):
    res = dict(np.load(out_path(a.run, a.it), allow_pickle=True))
    hdr = "  ".join(f"{lo:g}-{hi:g}" if hi < 1e8 else f">{lo:g}" for lo, hi in BANDS)
    for _, tag, _ in SPINS:
        if f"im_{tag}_base" not in res:
            continue
        rows, kf, w = res[f"rows_{tag}"], float(res[f"kf_{tag}"]), res[f"w_{tag}"]
        base = res[f"im_{tag}_base"]
        print(f"\n=== Sigma_{tag}  (kF = {kf:.4f})   scarto relativo L1 per banda di |omega|: {hdr}")
        names = ["prod"] + [n for n in VARIANTS if n != "base" and f"im_{tag}_{n}" in res]
        for i, kv in enumerate(rows):
            print(f"  riga k = {kv:.4f} ({kv / kf:.3f} kF)")
            for n in names:
                x = res[f"prod_{tag}"][i] if n == "prod" else res[f"im_{tag}_{n}"][i]
                e = band_err(w, base[i], x, signed=a.signed)
                lab = "base vs produzione" if n == "prod" else n
                print(f"     {lab:20s} " + "  ".join(f"{v:9.2e}" for v in e))
    return 0


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("compute", "report"))
    ap.add_argument("--run", default="P0p50_final")
    ap.add_argument("--it", type=int, default=8)
    ap.add_argument("--base", default=os.path.join(HERE, "out", "cluster"))
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--signed", action="store_true",
                    help="report: scarto con segno sum(var - base)/sum|base| (ImSigma < 0: positivo = |ImSigma| cala)")
    a = ap.parse_args(argv)
    return compute(a) if a.mode == "compute" else report(a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
