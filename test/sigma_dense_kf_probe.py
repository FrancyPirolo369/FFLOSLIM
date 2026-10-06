#!/usr/bin/env python3
"""N(omega) senza l'artefatto di interpolazione: Sigma calcolata su righe k fitte attorno a kF.

La pipeline calcola ImSigma solo su sigma_nk = 24 righe k e interpola (PCHIP) in mezzo: vicino
a kF le righe stanno a +-0.2-0.35 kF, e la cuspide non-FL della riga a kF finisce copiata in
tutto il segmento (lo spike di N(omega) a omega = 0, test/dos_spike_mechanism.py).  Qui, sullo
stato convergito di una run (cube di next_cubes + tabella di coppia della snap della stessa
iterazione), si calcola ImSigma con lo STESSO motore di density (memory_safe_im_sigma, stesse
variabili SIGMA_PN_* di run_test) su:
  A  le righe di produzione vicine a kF (dalla snap sigmaNNN, ck_*_k_sparse)
  B  le stesse + righe a kF (1 +- d), d = 0.005 ... 0.15
con gli stessi nodi omega (quelli di produzione con |omega| <= 2 + nodi fitti attorno a 0).
ReSigma per riga con la KK legacy di density (kk_re_pv_linear su w_base; oltre |omega| = 2
l'ImSigma del cubo, liscia in k a quelle frequenze), sigma0 = ReSigma(kF, 0), e N(omega) = int k dk A(k, omega) nella finestra di righe.
L'unica differenza fra A e B e' la densita' di righe in k: quello che cambia e' l'artefatto.

Uso (dalla radice di SLIM):
  python3 test/sigma_dense_kf_probe.py compute [--run P0p50_prod] [--workers 8]   (Sigma, lento)
  python3 test/sigma_dense_kf_probe.py plot    [--run P0p50_prod]                 (DOS, veloce)
  python3 test/sigma_dense_kf_probe.py time                                       (stima tempi)
"""
from __future__ import annotations

import os

# le stesse manopole di Sigma che test/run_test.py mette nell'ambiente (ricetta PROD)
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

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from fflo.density import load_seed, memory_safe_im_sigma  # noqa: E402
from fflo.kramers_kronig import kk_re_pv_linear  # noqa: E402
from fflo.pair_shifted import apply_shift_to_pair_table  # noqa: E402
from fflo.sigma_engine import Physics, load_cube_sigma  # noqa: E402

DELTA = 1.0e-3
W_UPDATE = (-216.0, 144.0)       # finestra di update di density (default)
N_OMEGA_PROD = 97                # sigma_nomega di PROD
W_FRESH = 2.0                    # ImSigma calcolata da zero per |omega| <= W_FRESH, oltre quella del cubo
FINE = np.array([0.001, 0.002, 0.004, 0.007, 0.01, 0.015, 0.02, 0.03, 0.045, 0.065, 0.09, 0.13])
DREL = np.array([0.01, 0.025, 0.05, 0.1, 0.15])
SPINS = (("down", "dn", "mu_dn"), ("up", "up", "mu_up"))


def paths(run, base):
    rd = os.path.join(base, run)
    cubes = sorted(glob.glob(os.path.join(rd, "iter*", "next_cubes", "A_komega_spinup_iter*.npz")))
    if not cubes:
        raise SystemExit(f"nessuna next_cubes in {rd}")
    up = cubes[-1]
    it = int(os.path.basename(up).split("iter")[-1][:3])
    return rd, it, up, up.replace("spinup", "spindown")


def omega_nodes(w_base):
    m = (w_base >= W_UPDATE[0] - 1e-12) & (w_base <= W_UPDATE[1] + 1e-12)
    wu = w_base[m]
    idx = np.unique(np.round(np.linspace(0, wu.size - 1, N_OMEGA_PROD)).astype(int))
    w = wu[idx]
    return np.unique(np.concatenate([w[np.abs(w) <= W_FRESH], FINE, -FINE]))


def setup(run, base):
    rd, it, up, dn = paths(run, base)
    seeds = {"up": load_seed(Path(up)), "down": load_seed(Path(dn))}
    z = np.load(os.path.join(rd, "snap", f"iter{it:03d}.npz"))
    qff = float(z["qff"])
    table = {"q": np.asarray(z["q"], float), "omega": np.asarray(z["omega"], float),
             "ReInvGamma": np.asarray(z["ReInvGamma"], float),
             "ImInvGamma": np.asarray(z["ImInvGamma"], float)}
    q_sel = float(table["q"][int(np.argmin(np.abs(table["q"] - qff)))])
    sh = apply_shift_to_pair_table(table, q_sel, subcritical_delta=DELTA, eta_floor_mode="exact_zero")
    s = np.load(os.path.join(rd, "snap", f"sigma{it:03d}.npz"), allow_pickle=True)
    phys = Physics(eps0=1.0, mu_up=seeds["up"]["mu_up"], mu_down=seeds["up"]["mu_dn"],
                   mass=seeds["up"]["mass"], pmf_scale=0.0)
    return dict(rd=rd, it=it, up=up, dn=dn, seeds=seeds, qff=qff, q_sel=q_sel,
                pq=np.asarray(sh["q"], float), pw=np.asarray(sh["omega"], float),
                pa=np.asarray(sh["A_pair"], float), snap_sigma=s, phys=phys)


def prod_omega_nodes(ctx, tag):
    """i nodi omega di Sigma della produzione (snap sigmaNNN: ck_*_omega)"""
    return np.asarray(ctx["snap_sigma"][f"ck_{tag}_omega"], float).ravel()


def rows_for(ctx, tag, mukey):
    kf = float(np.sqrt(ctx["seeds"]["up"]["mass"] * ctx["seeds"]["up"][mukey]))
    sparse = np.asarray(ctx["snap_sigma"][f"ck_{tag}_k_sparse"], float).ravel()
    lo, hi = kf - 0.4, kf + 0.5
    prod = sparse[(sparse >= lo) & (sparse <= hi)]
    drel = ctx.get("drel", DREL)
    dense = np.concatenate([kf * (1 - drel), [kf], kf * (1 + drel)])
    return kf, prod, np.unique(np.round(np.concatenate([prod, dense]), 10))


def compute(args):
    ctx = setup(args.run, args.base)
    if args.drel:
        ctx["drel"] = np.array([float(x) for x in args.drel.split(":")])
    out = os.path.join(HERE, "out", "sigma_dense_kf", args.run + (f"_{args.tag}" if args.tag else ""))
    os.makedirs(out, exist_ok=True)
    w_base = ctx["seeds"]["up"]["w_base"]
    w_eval = omega_nodes(w_base)
    save = {"w_eval": w_eval, "w_base": w_base, "it": ctx["it"], "qff": ctx["qff"]}
    for spin, tag, mukey in SPINS:
        if args.prod_omega:
            w_eval = prod_omega_nodes(ctx, tag)
            save[f"w_eval_{tag}"] = w_eval
        kf, prod, rows = rows_for(ctx, tag, mukey)
        if args.max_rows:
            rows = rows[np.argsort(np.abs(rows - kf))[:args.max_rows]]
            rows.sort()
        print(f"[{tag}] kF={kf:.6f}: {rows.size} righe ({prod.size} di produzione) x {w_eval.size} omega",
              flush=True)
        t0 = time.time()
        im = memory_safe_im_sigma(
            spin=spin, cube=ctx["seeds"]["down" if spin == "up" else "up"],
            pair_q=ctx["pq"], pair_omega=ctx["pw"], pair_a=ctx["pa"],
            k_target=rows,
            omega_target=(w_eval if not args.max_omega
                          else np.sort(w_eval[np.argsort(np.abs(w_eval))[:args.max_omega]])),
            physics=ctx["phys"], qff=ctx["q_sel"], sigma_nk=rows.size,
            sigma_k_feature_fraction=0.0, n_theta=32, omega_chunk=1000, workers=args.workers,
            checkpoint=Path(out) / f"imsigma_{tag}_ckpt.npz")
        print(f"[{tag}] fatto in {time.time() - t0:.0f} s", flush=True)
        save.update({f"kf_{tag}": kf, f"rows_{tag}": rows, f"prod_{tag}": prod, f"im_{tag}": im})
        if args.time_only:
            dt = time.time() - t0
            n = rows.size * (args.max_omega or w_eval.size)
            print(f"   {dt / n * args.workers:.2f} s-core per valutazione (k, omega)")
            return 0
    np.savez_compressed(os.path.join(out, "rows.npz"), **save)
    print("scritto", os.path.join(out, "rows.npz"))
    return 0


def sigma_rows(ctx, w_base, w_eval, rows, im_eval, cube_path):
    """ImSigma e ReSigma su w_base per ogni riga, come density: PCHIP in omega dentro la finestra,
    ImSigma del cubo fuori, KK legacy."""
    from scipy.interpolate import PchipInterpolator
    kc = np.load(cube_path)["k"]
    re_c, im_c = load_cube_sigma(Path(cube_path))
    inside = (w_base >= w_eval[0]) & (w_base <= w_eval[-1])
    im_full = np.empty((rows.size, w_base.size))
    re_full = np.empty_like(im_full)
    for i, kv in enumerate(rows):
        j = np.clip(np.searchsorted(kc, kv), 1, kc.size - 1)
        t = (kv - kc[j - 1]) / (kc[j] - kc[j - 1])
        row = (1 - t) * im_c[j - 1] + t * im_c[j]
        row[inside] = PchipInterpolator(w_eval, im_eval[i], extrapolate=False)(w_base[inside])
        im_full[i] = np.minimum(row, 0.0)
        re_full[i] = kk_re_pv_linear(w_base, im_full[i])
    return re_full, im_full


def dos(k_rows, re_rows, im_rows, w_base, w, kf, mu, eta, kk):
    from scipy.interpolate import PchipInterpolator
    re_w = np.array([np.interp(w, w_base, r) for r in re_rows])
    im_w = np.array([np.interp(w, w_base, r) for r in im_rows])
    re0 = np.array([np.interp(0.0, w_base, r) for r in re_rows])
    sigma0 = float(PchipInterpolator(k_rows, re0)(kf))
    re_k = PchipInterpolator(k_rows, re_w, axis=0)(kk)          # (nk, nw)
    im_k = PchipInterpolator(k_rows, im_w, axis=0)(kk)
    g = np.abs(np.minimum(im_k, 0.0)) + eta
    a = g / np.pi / ((w[None, :] - (kk[:, None] ** 2 - mu) - (re_k - sigma0)) ** 2 + g ** 2)
    return np.trapezoid(a * kk[:, None], kk, axis=0), sigma0


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ctx = setup(args.run, args.base)
    out = os.path.join(HERE, "out", "sigma_dense_kf", args.run + (f"_{args.tag}" if args.tag else ""))
    r = np.load(os.path.join(out, "rows.npz"))
    w_base = r["w_base"]
    w = np.linspace(-0.12, 0.12, 961)
    fig, axs = plt.subplots(3, 2, figsize=(12.5, 12.2), layout="constrained")
    for col, (spin, tag, mukey) in enumerate(SPINS):
        w_eval = r[f"w_eval_{tag}"] if f"w_eval_{tag}" in r.files else r["w_eval"]
        kf, rows, prod, im = float(r[f"kf_{tag}"]), r[f"rows_{tag}"], r[f"prod_{tag}"], r[f"im_{tag}"]
        mu = float(ctx["seeds"]["up"][mukey])
        cube = ctx["up"] if spin == "up" else ctx["dn"]
        eta = float(ctx["seeds"]["up"]["eta"])
        re_full, im_full = sigma_rows(ctx, w_base, w_eval, rows, im, cube)
        kk = np.linspace(rows[0], rows[-1], 9001)
        free = np.trapezoid(eta / np.pi / ((w[None, :] - (kk[:, None] ** 2 - mu)) ** 2 + eta ** 2) * kk[:, None], kk, axis=0)
        sel_a = np.isin(np.round(rows, 10), np.round(prod, 10))
        if not np.any(np.isclose(rows[sel_a], kf, atol=2e-3)):
            sel_a |= np.isclose(rows, kf)
        n_a, s0a = dos(rows[sel_a], re_full[sel_a], im_full[sel_a], w_base, w, kf, mu, eta, kk)
        n_b, s0b = dos(rows, re_full, im_full, w_base, w, kf, mu, eta, kk)
        # n(k) vicino a kF: int_{-2}^{0} A dw su una griglia fitta verso 0 (la parte a w < -2 e' liscia in k)
        wn = np.concatenate([-np.geomspace(2.0, 1e-7, 3000), [0.0]])
        kn = kf * np.linspace(0.94, 1.06, 481)
        from scipy.interpolate import PchipInterpolator as _P
        for lab, sel, colr in (("produzione", sel_a, "#1f1f1e"), ("righe fitte", np.ones(rows.size, bool), "#2a78d6")):
            re_w = np.array([np.interp(wn, w_base, x) for x in re_full[sel]])
            im_w = np.array([np.interp(wn, w_base, x) for x in im_full[sel]])
            s0 = float(_P(rows[sel], np.array([np.interp(0.0, w_base, x) for x in re_full[sel]]))(kf))
            re_k = _P(rows[sel], re_w, axis=0)(kn)
            im_k = _P(rows[sel], im_w, axis=0)(kn)
            g = np.abs(np.minimum(im_k, 0.0)) + eta
            a = g / np.pi / ((wn[None, :] - (kn[:, None] ** 2 - mu) - (re_k - s0)) ** 2 + g ** 2)
            nk_low = np.trapezoid(a, wn, axis=1)
            axs[2, col].plot(kn / kf, nk_low, ".-", ms=2, lw=1.2, color=colr, label=lab)
        axs[2, col].set(xlabel="k/kF", ylabel="int_{-2}^0 A dw",
                        title=f"n{'↓' if tag == 'dn' else '↑'}(k) vicino a kF (parte a |w| < 2)")
        axs[2, col].legend(fontsize=8)
        axs[2, col].grid(alpha=0.3)
        ax = axs[0, col]
        ax.plot(w, free, color="#9a9a96", lw=1, ls="--", label="liberi (stessa finestra in k)")
        ax.plot(w, n_a, color="#1f1f1e", lw=1.6, label=f"righe di produzione ({sel_a.sum()} nella finestra)")
        ax.plot(w, n_b, color="#2a78d6", lw=2, label=f"righe fitte a kF ({rows.size})")
        ax.set(title=f"N({'↓' if tag == 'dn' else '↑'}) — {args.run} it{int(r['it'])}", xlabel="ω")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        i0 = int(np.argmin(np.abs(w)))
        side = lambda n: 0.5 * (np.interp(-0.02, w, n) + np.interp(0.02, w, n))
        print(f"[{tag}] produzione: N(0) = {n_a[i0]:.4f}, spike = {n_a[i0] - side(n_a):+.4f} | "
              f"righe fitte: N(0) = {n_b[i0]:.4f}, spike = {n_b[i0] - side(n_b):+.4f} | "
              f"liberi {free[i0]:.4f} | sigma0 {s0a:+.5f} / {s0b:+.5f}")
        # ImSigma vicino a omega = 0 per le righe fitte: la cuspide sta solo a kF?
        ax = axs[1, col]
        cmap = plt.get_cmap("coolwarm")
        m = (w_eval > 0) & (w_eval < 0.14)
        mn = (w_eval < 0) & (w_eval > -0.14)
        near = rows[np.abs(rows / kf - 1) <= 0.1 + 1e-9]
        for kv in near:
            i = int(np.argmin(np.abs(rows - kv)))
            c = cmap(0.5 + (kv / kf - 1) / 0.2)
            ax.loglog(w_eval[m], -im[i][m], color=c, lw=1.4 if np.isclose(kv, kf) else 0.9,
                      label=f"k/kF = {kv / kf:.3f}" if abs(kv / kf - 1) in (0, 0.01, 0.05, 0.1) or np.isclose(kv, kf) else None)
            ax.loglog(-w_eval[mn], -im[i][mn], color=c, lw=0.8, ls=":")
        x = np.array([2e-3, 0.1])
        ik = int(np.argmin(np.abs(rows - kf)))
        y0 = -np.interp(0.02, w_eval, im[ik])
        ax.loglog(x, y0 * (x / 0.02) ** (2 / 3), color="#9a9a96", ls="--", lw=1, label="|ω|^(2/3)")
        ax.loglog(x, y0 * (x / 0.02) ** 2, color="#9a9a96", ls="-.", lw=1, label="ω²")
        ax.set(xlabel="|ω|  (continua ω>0, punteggiata ω<0)", ylabel="−ImΣ(k, ω)",
               title=f"ImΣ{'↓' if tag == 'dn' else '↑'} sulle righe vicine a kF")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3, which="both")
    dst = os.path.join(HERE, "out", "cluster", "plots",
                       f"dos_dense_kf_{args.run}{('_' + args.tag) if args.tag else ''}.png")
    fig.savefig(dst, dpi=115)
    print("scritto", dst)
    return 0


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("compute", "plot", "time"))
    ap.add_argument("--run", default="P0p50_prod")
    ap.add_argument("--base", default=os.path.join(HERE, "out", "cluster"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--max-omega", type=int, default=0)
    ap.add_argument("--drel", default="", help="offset relativi delle righe fitte, separati da ':' (default DREL)")
    ap.add_argument("--prod-omega", action="store_true", help="nodi omega di Sigma della produzione (snap)")
    ap.add_argument("--tag", default="", help="suffisso dei file di uscita")
    a = ap.parse_args(argv)
    a.time_only = a.mode == "time"
    if a.time_only:
        a.max_rows, a.max_omega = a.max_rows or 8, a.max_omega or 8
        return compute(a)
    return compute(a) if a.mode == "compute" else plot(a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
