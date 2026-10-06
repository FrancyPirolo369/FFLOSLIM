#!/usr/bin/env python3
"""Diagramma di fase con le run FINAL in corso: punti attuali, punto fisso stimato, linea vecchia.

Per ogni out/cluster/P0pXX_<--family> (default final), da tutte le snapshot:
    g_n = -4 pi max[ReG^-1(qff,0), ReG^-1(Q0+,0)] - ln2/2      (due candidati, come qff-or-zero)
    punto fisso stimato (Aitken sugli ultimi incrementi):  g* = g_n + dg_n r / (1 - r)
        r = media degli ultimi due rapporti dg_n/dg_(n-1), tagliata a --rmax;
        barra = meta' dello scarto fra le stime con l'ultimo e il penultimo rapporto
        (con un solo rapporto: barra = meta' del resto stimato).
La linea vecchia (bozza 2026-10-02: --old-families, --two-cand, min 14 iterazioni) e' in grigio;
le P che FINAL non ha ancora restano solo grigie.  Campo medio e NSCT come in plot_phase_diagram.py.

Uso (dalla radice di SLIM):  python3 test/plot_phase_final.py [--out out/cluster/plots/phase_diagram_final.png]
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_phase_diagram as ppd  # noqa: E402  (stile, collect, costanti)
import matplotlib.pyplot as plt  # noqa: E402

G0 = -0.5 * math.log(2.0)


def g_two(z):
    """g_c dallo shift davvero applicato da density (shift_used: riga qff, riga Q0+ o, con
    qff-or-zero-fit, il valore estrapolato dell'altopiano), e il margine Q0+ - qff in unita' di g."""
    if "shift_used" in z.files:
        q = np.asarray(z["q"], float)
        re0 = np.asarray(z["ReInvGamma"], float)[:, int(np.argmin(np.abs(np.asarray(z["omega"], float))))]
        qff = float(z["qff"])
        iq = int(np.argmin(np.abs(q - qff)))
        nz = np.flatnonzero(q > 1e-9)
        i0 = int(nz[np.argmin(q[nz])])
        return -4.0 * math.pi * float(z["shift_used"]) + G0, 4.0 * math.pi * (re0[i0] - re0[iq])
    q = np.asarray(z["q"], float)
    re0 = np.asarray(z["ReInvGamma"], float)[:, int(np.argmin(np.abs(np.asarray(z["omega"], float))))]
    qff = float(z["qff"])
    iq = int(np.argmin(np.abs(q - qff)))
    nz = np.flatnonzero(q > 1e-9)
    i0 = int(nz[np.argmin(q[nz])])
    return -4.0 * math.pi * max(re0[iq], re0[i0]) + G0, 4.0 * math.pi * (re0[i0] - re0[iq])


def p_effective(run_dir, P, eta=1.0e-3, wmin=-630.0, wmax=630.0):
    """P misurata all'ultima iterazione: densita' della cube mescolata (dr_alpha_0.3_density_*) meno
    la coda lorentziana eta/(pi xi) del picco QP (finta occupazione a w < 0, solo contabilita')."""
    snaps = sorted(glob.glob(os.path.join(run_dir, "snap", "sigma*.npz")))
    if not snaps:
        return float("nan")
    s = np.load(snaps[-1], allow_pickle=True)
    out = {}
    for tag, sp, mu in (("dn", "down", 1.0 - P), ("up", "up", 1.0 + P)):
        key = f"dr_alpha_0.3_density_{sp}"
        if key not in s.files:
            return float("nan")
        k = np.asarray(s[f"k_{tag}"], float)
        xi = k ** 2 - mu
        gain = np.where(xi > 3 * eta, eta / np.pi * (1 / np.maximum(xi, 1e-9) - 1 / (xi - wmin)), 0.0)
        loss = np.where(xi < -3 * eta, eta / np.pi * (1 / np.maximum(-xi, 1e-9) - 1 / (wmax - xi)), 0.0)
        out[tag] = float(s[key]) - np.trapezoid(k * (gain - loss), k) / 2.0
    return (out["up"] - out["dn"]) / (out["up"] + out["dn"])


def aitken(gs, rmax):
    dg = np.diff(gs)
    if dg.size < 2:
        return float("nan"), float("nan"), []
    rr = [dg[i + 1] / dg[i] for i in range(dg.size - 1) if dg[i] != 0.0]
    if not rr or any(not (0.0 < r < 1.0) for r in rr[-2:]):
        # traiettoria che ha invertito o accelera (secondo modo lento, 2026-10-02): niente estrapolazione
        return float("nan"), float("nan"), rr
    clip = [min(r, rmax) for r in rr[-2:]]
    est = [gs[-1] + dg[-1] * r / (1.0 - r) for r in clip]
    rbar = float(np.mean(clip))
    gstar = gs[-1] + dg[-1] * rbar / (1.0 - rbar)
    err = 0.5 * abs(est[-1] - est[0]) if len(est) == 2 else 0.5 * abs(gstar - gs[-1])
    return gstar, err, rr


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=os.path.join(ppd.HERE, "out", "cluster"))
    ap.add_argument("--family", default="final")
    ap.add_argument("--old-families", default="prod,gmax,x75a0p3,prod_union_fine")
    ap.add_argument("--rmax", type=float, default=0.93)
    ap.add_argument("--smooth", type=float, default=0.02)
    ap.add_argument("--peff", action="store_true",
                    help="punti alla P effettiva misurata (densita' della cube mescolata, coda eta tolta) "
                         "invece che alla P nominale; una linea sottile li collega alla nominale")
    ap.add_argument("--out", default=os.path.join(ppd.HERE, "out", "cluster", "plots", "phase_diagram_final.png"))
    a = ap.parse_args(argv)
    from scipy.interpolate import UnivariateSpline

    old = ppd.collect(a.base, [f.strip() for f in a.old_families.split(",")], False, 14, 0.1, True)
    new = {}
    for d in sorted(glob.glob(os.path.join(a.base, f"P0p*_{a.family}"))):
        p = float("0." + re.search(r"P0p(\d+)_", os.path.basename(d) + "_").group(1))
        snaps = sorted(glob.glob(os.path.join(d, "snap", "iter*.npz")))
        if not snaps:
            continue
        seq = [g_two(np.load(s)) for s in snaps]
        gs = np.array([x[0] for x in seq])
        gstar, err, rr = aitken(gs, a.rmax)
        # ramo dal pin che density ha davvero usato (riga PAIR di loop.log, Q=): Q < 0.1 qff = pBCS
        qff = float(np.load(snaps[-1])["qff"])
        qpin = [float(m) for m in re.findall(r"PAIR: .* Q=([0-9.eE+-]+)", open(os.path.join(d, "loop.log")).read())]
        branches = ["pBCS" if v < 0.1 * qff else "FFLO" for v in qpin]
        phase = branches[-1] if branches else "FFLO"
        contested = len(set(branches[-3:])) > 1
        if phase == "pBCS" or contested or len(gs) < 5:
            gstar, err = float("nan"), float("nan")      # con meno di 5 iterazioni la stima non regge
        new[p] = dict(it=len(gs), g=gs[-1], gstar=gstar, err=err, rr=rr, marg=seq[-1][1],
                      peff=p_effective(d, p),
                      dg=gs[-1] - gs[-2] if len(gs) > 1 else float("nan"), phase=phase,
                      contested=contested, branches="".join("F" if b == "FFLO" else "B" for b in branches[-6:]))

    print(f"{'P':>5s}  {'vecchia':>8s}  {'final it':>8s}  {'g ora':>8s}  {'ultimo dg':>9s}  {'r':>11s}  {'g* stimato':>16s}  margine Q~0  ramo (ultime it: F=qff, B=Q~0)")
    for p in sorted(set(old) | set(new)):
        o = f"{old[p]['g']:+.3f}" if p in old else "-"
        if p in new:
            n = new[p]
            print(f"{p:5.2f}  {o:>8s}  {n['it']:8d}  {n['g']:+8.4f}  {n['dg']:+9.4f}  "
                  f"{' '.join(f'{r:.2f}' for r in n['rr'][-2:]):>11s}  {n['gstar']:+8.3f} +- {n['err']:.3f}  {n['marg']:+.3f}"
                  f"       {n['phase']}{' CONTESO' if n['contested'] else ''} ({n['branches']})  P_eff={n['peff']:.4f}")
        else:
            print(f"{p:5.2f}  {o:>8s}  {'-':>8s}")

    fig, ax = plt.subplots(figsize=(8.6, 6.4), layout="constrained")
    eps0 = np.geomspace(1e-4, 1.0, 600)
    ax.plot(0.5 * np.log(eps0 / 2.0), eps0 * np.sqrt(2.0 / eps0 - 1.0), "-", color=ppd.C_MF, lw=1.2,
            label="mean-field/psct")
    nsct = os.path.join(ppd.HERE, "test", "data", "nsct_critical_line.txt")
    if os.path.exists(nsct):
        d = np.loadtxt(nsct)
        d = d[np.argsort(d[:, 2])]
        cut = np.flatnonzero(np.diff(d[:, 2]) > 0.5) + 1
        for i, br in enumerate(np.split(d, cut)):
            ax.plot(br[:, 2], br[:, 3], "-.", color=ppd.C_NSCT, lw=1.2, label="NSCT" if i == 0 else None)

    po = sorted(old)
    go = np.array([old[p]["g"] for p in po])
    if len(po) >= 4:
        spl = UnivariateSpline(po, go, k=3, s=len(po) * a.smooth ** 2)
        pf = np.linspace(min(po), max(po), 300)
        ax.plot(spl(pf), pf, "-", color="#b5b4af", lw=1.6, label="riferimento: " + a.old_families.replace(",", ", ") + " (ricetta prod)")
    ax.plot(go, po, "s", color="#b5b4af", ms=5.5, mec="#fcfcfb", mew=1.0)

    # ordinata: P nominale, oppure (--peff) la P effettiva misurata
    Y = {p: (new[p]["peff"] if a.peff and np.isfinite(new[p]["peff"]) else p) for p in new}
    if a.peff:
        for p in new:
            if abs(Y[p] - p) > 1e-4:
                ax.plot([new[p]["g"], new[p]["g"]], [p, Y[p]], "-", color=ppd.MUTED, lw=0.8, alpha=0.7)
                ax.plot([new[p]["g"]], [p], "_", color=ppd.MUTED, ms=8)
    pn = sorted((p for p in new if new[p]["phase"] == "FFLO" and not new[p]["contested"]), key=lambda p: Y[p])
    gn = np.array([new[p]["g"] for p in pn])
    yn = np.array([Y[p] for p in pn])
    if len(pn) >= 4:
        spl = UnivariateSpline(yn, gn, k=3, s=len(pn) * a.smooth ** 2)
        pf = np.linspace(min(yn), max(yn), 300)
        ax.plot(spl(pf), pf, "-", color=ppd.C_NEW, lw=2.0, label=f"{a.family.upper()}, ramo FFLO (pin a qff)")
    ax.plot(gn, yn, "s", color=ppd.C_NEW, ms=7, mec="#fcfcfb", mew=1.2)
    pb = [p for p in new if new[p]["phase"] == "pBCS" and not new[p]["contested"]]
    if pb:
        ax.plot([new[p]["g"] for p in pb], [Y[p] for p in pb], "^", color=ppd.C_PBCS, ms=8, mec="#fcfcfb", mew=1.2,
                label=f"{a.family.upper()}, ramo pBCS (pin a Q ~ 0)")
    pc = [p for p in new if new[p]["contested"]]
    if pc:
        ax.plot([new[p]["g"] for p in pc], [Y[p] for p in pc], "D", color="#fcfcfb", mec=ppd.C_PBCS, mew=1.6, ms=7,
                label=f"{a.family.upper()}, canale conteso (salta fra qff e Q ~ 0)")
    pn = sorted(new)
    first = True
    for p in pn:
        n = new[p]
        ax.annotate(f"it {n['it']}", (n["g"], Y[p]), textcoords="offset points", xytext=(0, -13),
                    fontsize=7, color=ppd.C_NEW, ha="center")
        if np.isfinite(n["gstar"]):
            ax.annotate("", xy=(n["gstar"], Y[p]), xytext=(n["g"], Y[p]),
                        arrowprops=dict(arrowstyle="->", color=ppd.C_NEW, lw=0.9, alpha=0.7))
            ax.errorbar([n["gstar"]], [Y[p]], xerr=[n["err"]], fmt="o", color=ppd.C_NEW, mfc="#fcfcfb",
                        ms=6.5, mew=1.4, elinewidth=1.0, capsize=2.5,
                        label=f"{a.family.upper()}, punto fisso stimato (Aitken)" if first else None)
            first = False
    for p in po:
        if p not in new:
            ax.annotate("non ancora in FINAL", (old[p]["g"], p), textcoords="offset points",
                        xytext=(8, -3), fontsize=7, color=ppd.MUTED)
    ax.text(0.99, 0.01, "punto fisso stimato solo con >= 5 iterazioni, traiettoria monotona e pin stabile; "
            "barra = scarto fra le stime con l'ultimo e il penultimo rapporto", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color=ppd.MUTED)
    ax.set(xlabel="g_c", ylabel=("P effettiva (densita' misurate, coda eta tolta)" if a.peff else "P"),
           xlim=(-3.6, 2.5), ylim=(0, 1))
    ax.legend(fontsize=8, loc="upper left")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print("scritto", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
