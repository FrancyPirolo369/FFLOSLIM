#!/usr/bin/env python3
"""Diagramma di fase: g_c in ascissa, P in ordinata, risultato nuovo contro FFLO30 (luglio).

Per ogni P prende l'ultima iterazione della prima famiglia disponibile, nell'ordine di
--families (default prodg, gmax, prod: prima il massimo globale, poi PROD a qff).
g_c = -4 pi shift - ln2/2, lo shift e' quello applicato da density (riga PAIR di loop.log).
Per le famiglie pinnate a qff (prod, etaexact) --gmax-corr aggiunge la correzione al primo
ordine per pinnare il massimo globale di ReGamma^-1(Q,0) (dall'ultima snapshot).
Sul risultato nuovo una spline cubica di smoothing g(P) (--smooth = scarto tipico atteso
per punto; 0 = interpolante).  --exclude toglie delle P (default 0.5: in overshoot).

Campo medio (T = 0, soglia FFLO): P_c = eps0 sqrt(2/eps0 - 1) = sqrt(eps0 (2 - eps0)), con
g = ln(eps0/2)/2 (la stessa relazione eps0 <-> g dei punti numerici), eps0 in (0, 1].

NSCT (T-matrix non autoconsistente): test/data/nsct_critical_line.txt, due rami (debole,
P <= 0.1 per g <= -0.8; forte, da g = 0.2 fino a P ~ 1).  Densita' fissata: la sua
normalizzazione di eps0 coincide con mu_avg = 1 solo ad accoppiamento debole.

FFLO30: r30b_pull/critical_line_status.txt (scansione r30 0.05-0.40 e continuazioni r30b
0.35-0.70, 9-12 iterazioni; stessa formula di g_c, eps0 = 1).  Il punto a P = 0.70 ha
ReGamma^-1(qff) < 0 (canale q ~ 0 gia' vincente): escluso salvo --fflo30-all.

Uso (dalla radice di SLIM):
  python3 test/plot_phase_diagram.py [--exclude 0.5] [--scan] [--out out/cluster/plots/phase_diagram.png]
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INK, MUTED, GRID = "#1f1f1e", "#6b6b68", "#e4e3df"
C_NEW, C_F30, C_SCAN, C_MF, C_NSCT = "#eb6834", "#2a78d6", "#6b6b68", "#1baf7a", "#e87ba4"
MARK = {"prodg": "D", "gmax": "^", "prod": "s", "etaexact": "o"}
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "legend.frameon": False, "lines.linewidth": 1.5,
})


def g_of(shift):
    return -4.0 * np.pi * shift - 0.5 * np.log(2.0)


def last_shift(loop_log):
    """(iterazione, g_c) dell'ultima riga PAIR, e la deriva di g_c per iterazione sulle ultime 3."""
    its, gs = [], []
    for line in open(loop_log):
        m = re.search(r"iter (\d+) PAIR: .*shift=([+-][\d.eE+-]+)", line)
        if m:
            its.append(int(m[1]))
            gs.append(g_of(float(m[2])))
    if not its:
        return None
    n = len(gs)
    drift = (gs[-1] - gs[max(0, n - 3)]) / max(min(2, n - 1), 1) if n > 1 else np.nan
    return its[-1], gs[-1], drift


def gmax_correction(run_dir):
    snaps = sorted(glob.glob(os.path.join(run_dir, "snap", "iter*.npz")))
    if not snaps:
        return 0.0
    z = np.load(snaps[-1])
    q, w = np.asarray(z["q"]), np.asarray(z["omega"])
    re0 = np.asarray(z["ReInvGamma"], float)[:, int(np.argmin(np.abs(w)))]
    iq = int(np.argmin(np.abs(q - float(z["qff"]))))
    return float(-4.0 * np.pi * (np.nanmax(re0) - re0[iq]))


def collect(base, families, gmax_corr, min_iters):
    pts = {}
    for fam in families:
        for d in sorted(glob.glob(os.path.join(base, f"P0p*_{fam}"))):
            m = re.search(r"P0p(\d+)_", os.path.basename(d) + "_")
            p = float("0." + m.group(1))
            if p in pts or not os.path.exists(os.path.join(d, "loop.log")):
                continue
            r = last_shift(os.path.join(d, "loop.log"))
            if r is None or r[0] < min_iters:
                continue
            corr = gmax_correction(d) if (gmax_corr and fam in ("prod", "etaexact")) else 0.0
            pts[p] = dict(p=p, fam=fam, it=r[0], g=r[1] + corr, drift=r[2], corr=corr)
    return pts


def read_fflo30(path, keep_all):
    out = {}
    for line in open(path):
        f = line.split()
        if not f or f[0] == "series":
            continue
        p, reinv, gc = float(f[1]), float(f[4]), float(f[5])
        if reinv < 0 and not keep_all:
            continue
        out[p] = gc if f[0].startswith("r30b") or p not in out else out[p]   # r30b prevale su r30
    return out


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=os.path.join(HERE, "out", "cluster"))
    ap.add_argument("--families", default="prodg,gmax,prod")
    ap.add_argument("--min-iters", type=int, default=1)
    ap.add_argument("--no-gmax-corr", action="store_true")
    ap.add_argument("--exclude", default="0.5", help="P da togliere, separate da virgole ('' = nessuna)")
    ap.add_argument("--smooth", type=float, default=0.03,
                    help="scarto tipico per punto della spline in g_c (0 = interpolante)")
    ap.add_argument("--scan", action="store_true", help="aggiunge la scansione SLIM (etaexact, it 30)")
    ap.add_argument("--fflo30", default=os.path.join(os.path.dirname(HERE), "r30b_pull",
                                                     "critical_line_status.txt"))
    ap.add_argument("--fflo30-all", action="store_true")
    ap.add_argument("--no-mean-field", action="store_true")
    ap.add_argument("--nsct", default=os.path.join(HERE, "test", "data", "nsct_critical_line.txt"),
                    help="file NSCT ('' = non disegnarlo)")
    ap.add_argument("--xlim", default="-3.6,2.0")
    ap.add_argument("--out", default=os.path.join(HERE, "out", "cluster", "plots", "phase_diagram.png"))
    a = ap.parse_args(argv)
    from scipy.interpolate import UnivariateSpline

    excl = {round(float(x), 4) for x in a.exclude.split(",") if x.strip()}
    pts = collect(a.base, [f.strip() for f in a.families.split(",")], not a.no_gmax_corr, a.min_iters)
    keep = [pts[p] for p in sorted(pts) if round(p, 4) not in excl]
    print(f"{'P':>5s}  famiglia  it   g_c     (corr. max globale)  deriva/it")
    for p in sorted(pts):
        r = pts[p]
        tag = "  ESCLUSO" if round(p, 4) in excl else ""
        print(f"{p:5.2f}  {r['fam']:8s} {r['it']:3d}  {r['g']:+.3f}   ({r['corr']:+.3f})          {r['drift']:+.4f}{tag}")

    fig, ax = plt.subplots(figsize=(8.2, 6.2), layout="constrained")
    if not a.no_mean_field:
        eps0 = np.geomspace(1e-4, 1.0, 600)
        ax.plot(0.5 * np.log(eps0 / 2.0), eps0 * np.sqrt(2.0 / eps0 - 1.0), "-", color=C_MF,
                lw=1.4, label="campo medio")
    if a.nsct and os.path.exists(a.nsct):
        d = np.loadtxt(a.nsct)
        d = d[np.argsort(d[:, 2])]
        cut = np.flatnonzero(np.diff(d[:, 2]) > 0.5) + 1          # rami separati da un buco in g
        for i, br in enumerate(np.split(d, cut)):
            ax.plot(br[:, 2], br[:, 3], "-.", color=C_NSCT, lw=1.5, label="NSCT" if i == 0 else None)
    if os.path.exists(a.fflo30):
        f30 = read_fflo30(a.fflo30, a.fflo30_all)
        ps = sorted(f30)
        ax.plot([f30[p] for p in ps], ps, "o--", color=C_F30, ms=4.5, lw=1.2,
                label="FFLO30, luglio (r30 + r30b, 9-12 it)")
    if a.scan:
        sc = collect(a.base, ["etaexact"], False, 1)
        ps = [p for p in sorted(sc) if p <= 0.7]
        ax.plot([sc[p]["g"] for p in ps], ps, "o:", color=C_SCAN, ms=4, lw=1.0,
                label="scansione SLIM (qff, it 30)")
    if len(keep) >= 4:
        P = np.array([r["p"] for r in keep]); G = np.array([r["g"] for r in keep])
        s = len(P) * a.smooth ** 2
        spl = UnivariateSpline(P, G, k=3, s=s)
        pf = np.linspace(P.min(), P.max(), 300)
        ax.plot(spl(pf), pf, "-", color=C_NEW, lw=2.2, label=f"spline sul risultato nuovo (±{a.smooth:g})")
        res = G - spl(P)
        print("scarti dalla spline: " + "  ".join(f"{p:.2f}:{x:+.3f}" for p, x in zip(P, res)))
    for fam in dict.fromkeys(r["fam"] for r in keep):
        rr = [r for r in keep if r["fam"] == fam]
        lab = {"prodg": "PROD al massimo globale", "gmax": "HI (massimo globale)",
               "prod": "PROD (qff)" + ("" if a.no_gmax_corr else " + correz. max globale")}[fam]
        ax.plot([r["g"] for r in rr], [r["p"] for r in rr], MARK.get(fam, "o"), color=C_NEW,
                ms=7, mec="#fcfcfb", mew=1.2, label=lab)
    for r in keep:
        ax.annotate(f"it {r['it']}", (r["g"], r["p"]), textcoords="offset points",
                    xytext=(7, -3), fontsize=7, color=MUTED)
    ymax = max([r["p"] for r in keep] + [0.7]) + 0.08
    x0, x1 = (float(v) for v in a.xlim.split(","))
    ax.set(xlabel="g_c", ylabel="P", ylim=(0, ymax), xlim=(x0, x1),
           title="Linea critica (Thouless) del gas di Fermi 2D polarizzato")
    ax.legend(fontsize=8, loc="upper left")
    if excl:
        ax.text(0.99, 0.01, "esclusi: P = " + ", ".join(f"{x:g}" for x in sorted(excl)),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=MUTED)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print("scritto", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
