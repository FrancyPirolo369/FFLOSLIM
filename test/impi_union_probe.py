#!/usr/bin/env python3
"""Parte a griglia di ImPi (A*A - A0*A0 di impi_table) con nodi in eps scelti per ogni Omega.

Riproduce l'integrale di fflo/impi_table.py (ricostruzione di A da Sigma, integrale angolare
phipanel, pannelli in eps con i nodi di Fermi, p lineare) su poche righe di Q:
  lattice : il reticolo in eps di produzione (build_omega_lattice su [-24, 160]); l'estremo
            mobile eps = Omega dell'integrale buca-buca e' inserito interpolando il prodotto
            fra i due nodi vicini (quello che fa impi_table).
  union   : reticolo + il suo nucleo fitto spostato su eps = Omega (livello di Fermi del
            minoritario, Omega - eps = 0) + il nodo eps = Omega, con l'integrando calcolato
            esattamente su ogni nodo.
Solo la parte a griglia: la parte QP x QP semi-analitica non dipende dal reticolo in eps.

Uso (dalla radice di SLIM):
  python3 test/impi_union_probe.py --up UP.npz --down DN.npz --a0-up A0_UP.npz --a0-down A0_DN.npz
      [--q 0,0.5,1 (in unita' di qff)] [--wmin -2 --wmax 0.4 --nw 961] [--tails 100,20]
      [--out out/impi_probe/NAME]
Senza cube usa quelle locali di out/smoke (P = 0.65).  Le A0 si fanno con
fflo.qp_cube.build_qp_cube (--make-a0 le costruisce accanto all'uscita).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import fflo.impi_table as T  # noqa: E402
from fflo.adn_phi import build_phipanel_geometry_cache, precompute_kjac_integrated_slice  # noqa: E402
from fflo.kramers_kronig import compute_impi_raw_totals_for_slice, finalize_impi_scan  # noqa: E402
from fflo.lattice_grids import build_omega_lattice  # noqa: E402

G: dict = {}
SMOKE = os.path.join(HERE, "out", "smoke")


def setup(up, down, a0up, a0dn, *, n_linear=40, n_tail=100, n_p=31, p_max=4.0, n_kquad=10,
          kquad_feat=7, core_half_width=0.12, dw=1e-3):
    ku, wu, _ = T.load_akw_cube(up)
    kd, wd, ad = T.load_akw_cube(down)
    cku, cwu, cau = T.load_akw_cube(a0up)
    ckd, cwd, cad = T.load_akw_cube(a0dn)
    re_up, im_up, s0u, etau = T._load_sigma_rebuild_fields(up)
    re_dn, im_dn, s0d, etad = T._load_sigma_rebuild_fields(down)
    _, mu_up, mu_dn = T.load_cube_physics(up)
    lat = build_omega_lattice(dw, n_linear, 160.0, -24.0, n_tail, n_tail)
    G.clear()
    G.update(ku=ku, wu=wu, kd=kd, wd=wd, ad=ad, cku=cku, cwu=cwu, cau=cau, ckd=ckd, cwd=cwd,
             cad=cad, re_up=re_up, im_up=im_up, s0u=s0u, etau=etau, re_dn=re_dn, im_dn=im_dn,
             s0d=s0d, etad=etad, mu_up=mu_up, mu_dn=mu_dn, lat=lat,
             core=lat[np.abs(lat) <= core_half_width], p=np.linspace(0.0, p_max, n_p),
             n_kquad=n_kquad, kquad_feat=kquad_feat,
             kf=[float(np.sqrt(mu_up)), float(np.sqrt(mu_dn))])
    return float(np.sqrt(mu_up) - np.sqrt(mu_dn))


def eps_nodes(omega, mode):
    lat = G["lat"]
    if mode == "lattice":
        return lat
    nodes = np.unique(np.concatenate([lat, G["core"] + omega, [omega]]))
    return nodes[(nodes >= lat[0]) & (nodes <= lat[-1])]


def _row(task):
    qv, omegas, mode = task
    p = G["p"]
    geom = build_phipanel_geometry_cache(G["kd"], p, q_value=float(qv), routing="difference",
                                         n_phi_panel=G["n_kquad"], phi_panel_chunk_size=4,
                                         k_features=G["kf"], k_feature_n_local=G["kquad_feat"],
                                         k_feature_half_width=0.05)
    out = np.zeros(len(omegas))
    up_cache = None
    for j, om in enumerate(omegas):
        eps = eps_nodes(float(om), mode)
        if mode == "lattice" and up_cache is not None:
            aup, cup = up_cache
        else:
            aup, inu = T._rebuild_a_bilinear(p[:, None], eps[None, :], G["ku"], G["wu"], G["re_up"],
                                             G["im_up"], mu=G["mu_up"], sigma0=G["s0u"], eta=G["etau"])
            cup, inc = T.bilinear_eval_with_support(p[:, None], eps[None, :], G["cku"], G["cwu"], G["cau"])
            aup, cup = np.where(inu, aup, 0.0), np.where(inc, cup, 0.0)
            if mode == "lattice":
                up_cache = (aup, cup)
        full, _ = precompute_kjac_integrated_slice(
            G["kd"], G["wd"], G["ad"], p, eps, q_value=float(qv), omega_value=float(om),
            routing="difference", n_kquad=G["n_kquad"], kquad_chunk_size=4, geom_cache=geom,
            re_sigma=G["re_dn"], im_sigma=G["im_dn"], sigma0=G["s0d"], mu=G["mu_dn"], eta=G["etad"])
        ctrl, _ = precompute_kjac_integrated_slice(
            G["ckd"], G["cwd"], G["cad"], p, eps, q_value=float(qv), omega_value=float(om),
            routing="difference", n_kquad=G["n_kquad"], kquad_chunk_size=4, geom_cache=geom)
        prod = aup * full - cup * ctrl
        th, nt, med = compute_impi_raw_totals_for_slice(float(om), p, eps, prod, np.ones_like(prod),
                                                        eps_integration_mode="panel_kinks",
                                                        p_integration_mode="plain")
        out[j] = finalize_impi_scan(np.array([om]), np.array([th]), np.array([nt]),
                                    np.array([med]))["im_pi_thermal"][0]
    return out


def run(qlist, omegas, mode, workers):
    n_chunks = max(1, (2 * workers) // max(1, len(qlist)))
    chunks = np.array_split(np.asarray(omegas, float), n_chunks)
    tasks = [(q, c, mode) for q in qlist for c in chunks]
    with mp.get_context("fork").Pool(workers) as pool:
        res = pool.map(_row, tasks)
    return np.array([np.concatenate(res[i * n_chunks:(i + 1) * n_chunks]) for i in range(len(qlist))])


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--up", default=os.path.join(SMOKE, "iter002/next_cubes/A_komega_spinup_iter002.npz"))
    ap.add_argument("--down", default=os.path.join(SMOKE, "iter002/next_cubes/A_komega_spindown_iter002.npz"))
    ap.add_argument("--a0-up", default=os.path.join(SMOKE, "iter003/pairbuild/control/A0_up.npz"))
    ap.add_argument("--a0-down", default=os.path.join(SMOKE, "iter003/pairbuild/control/A0_down.npz"))
    ap.add_argument("--make-a0", action="store_true", help="costruisce le A0 con build_qp_cube in --out")
    ap.add_argument("--q", default="0,0.5,1", help="righe di Q in unita' di qff")
    ap.add_argument("--wmin", type=float, default=-2.0)
    ap.add_argument("--wmax", type=float, default=0.4)
    ap.add_argument("--nw", type=int, default=961)
    ap.add_argument("--tails", default="100", help="code del reticolo in eps da provare")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "impi_probe", "smoke_P0p65"))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    a0u, a0d = a.a0_up, a.a0_down
    if a.make_a0:
        from fflo.qp_cube import build_qp_cube, qp_model_from_cube
        a0u, a0d = os.path.join(a.out, "A0_up.npz"), os.path.join(a.out, "A0_down.npz")
        for src, dst in ((a.up, a0u), (a.down, a0d)):
            if not os.path.exists(dst):
                build_qp_cube(src, dst, model=qp_model_from_cube(src))
    w = np.linspace(a.wmin, a.wmax, a.nw)
    res = {}
    for tail in [int(x) for x in a.tails.split(",")]:
        qff = setup(a.up, a.down, a0u, a0d, n_tail=tail)
        q = [float(x) * qff for x in a.q.split(",")]
        for mode in ("lattice", "union"):
            t0 = time.time()
            res[f"{mode}{tail}"] = run(q, w, mode, a.workers)
            print(f"{mode}{tail}: {time.time() - t0:.0f}s", flush=True)
    np.savez(os.path.join(a.out, "impi_union_probe.npz"), q=np.array(q), qff=qff, w=w,
             up=a.up, down=a.down, **res)
    plot(os.path.join(a.out, "impi_union_probe.npz"))
    return 0


def plot(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = np.load(path)
    q, w, qff = z["q"], z["w"], float(z["qff"])
    tails = sorted({int(k[5:]) for k in z.files if k.startswith("union")}, reverse=True)
    fig, ax = plt.subplots(2, q.size, figsize=(5.2 * q.size, 7), layout="constrained", squeeze=False)
    cols = {"lattice": "#2a78d6", "union": "#eb6834"}
    for i in range(q.size):
        for tail in tails:
            ls = "-" if tail == tails[0] else ":"
            for mode in ("lattice", "union"):
                ax[0, i].plot(w, z[f"{mode}{tail}"][i] * 1e3, ls, color=cols[mode], lw=1.0,
                              label=f"{mode} {tail}")
            ax[1, i].plot(w, (z[f"lattice{tail}"][i] - z[f"union{tail}"][i]) * 1e3, ls, color="#1f1f1e",
                          lw=0.9, label=f"lattice {tail} − union {tail}")
        ax[0, i].set_title(f"Q = {q[i] / qff:.2f} qff", fontsize=10)
        ax[1, i].set_xlabel("Ω")
        for r in (0, 1):
            ax[r, i].axvline(0, color="#6b6b68", lw=0.6)
    ax[0, 0].set_ylabel("ImΠ, parte a griglia  ×10⁻³")
    ax[1, 0].set_ylabel("differenza  ×10⁻³")
    ax[0, 0].legend(fontsize=8, frameon=False)
    ax[1, 0].legend(fontsize=8, frameon=False)
    out = path.replace(".npz", ".png")
    fig.savefig(out, dpi=120)
    print("scritto", out)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
