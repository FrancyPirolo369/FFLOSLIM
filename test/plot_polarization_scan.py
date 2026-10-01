#!/usr/bin/env python3
"""Diagramma riassuntivo dello scan in polarizzazione di test/submit.sh P.

Legge soltanto ``config.json``, ``loop.log`` e l'ultimo ``snap/iter*.npz``
scaricato dal cluster.  Non servono cube, Sigma o tabelle pairbuild complete.

Uso dalla radice di X_FFLO_SLIM::

    python3 test/plot_polarization_scan.py [BASE]

BASE predefinita: out/cluster/P_scan.  Scrive ``plots/phase_diagram.png``,
``plots/gamma_competition.png`` e ``plots/phase_diagram.csv``.

Il grafico descrive la linea ripinnata a Thouless, non un diagramma
termodinamico a epsilon_0 fisso: ogni P ha uno shift (e quindi un
epsilon_0_eff) diverso.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


DELTA = 1.0e-3
PAIR_RE = re.compile(
    r"=== iter\s+(\d+) PAIR: contact=([+\-0-9.eE]+).*?shift=([+\-0-9.eE]+)"
)
GAP_RE = re.compile(
    r"=== iter\s+(\d+) GAP: up=([+\-0-9.eE]+)% down=([+\-0-9.eE]+)%"
)
TAG_RE = re.compile(r"^P0p(\d+)_etaexact$")

INK = "#1f1f1e"
MUTED = "#6b6b68"
GRID = "#e4e3df"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#1baf7a"
RED = "#e34948"
PURPLE = "#4a3aa7"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
    "legend.frameon": False,
    "lines.linewidth": 1.5,
})


def log_series(path):
    pair, gap = {}, {}
    with open(path) as fh:
        for line in fh:
            m = PAIR_RE.search(line)
            if m:
                pair[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
            m = GAP_RE.search(line)
            if m:
                gap[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    common = sorted(set(pair) & set(gap))
    if not common:
        raise RuntimeError(f"nessuna iterazione completa in {path}")
    return pair, gap, common


def last_snapshot(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "snap", "iter*.npz")))
    if not files:
        raise RuntimeError(f"snapshot assente in {run_dir}")
    return files[-1]


def load_run(run_dir, p, control=False):
    pair, gap, common = log_series(os.path.join(run_dir, "loop.log"))
    it = common[-1]
    contact, shift = pair[it]
    gap_up, gap_dn = gap[it]
    history = np.array([[j, pair[j][0], pair[j][1]] for j in sorted(pair)], float)

    with np.load(last_snapshot(run_dir)) as z:
        q = np.asarray(z["q"], float)
        omega = np.asarray(z["omega"], float)
        re0 = np.asarray(z["ReInvGamma"], float)[:, int(np.argmin(np.abs(omega)))]
        qff = float(z["qff"])

    # run_test seleziona esattamente il nodo qff; l'accoppiamento applicato
    # sottrae shift + DELTA.  Quindi qff e' sempre a -DELTA, mentre il massimo
    # globale puo' essere stabile (<0) oppure supercritico (>0).
    iq = int(np.nanargmax(re0))
    q_star = float(q[iq])
    re_qff = float(np.interp(qff, q, re0))
    max_shifted_delta = float((re0[iq] - shift - DELTA) / DELTA)
    advantage_delta = float((re0[iq] - re_qff) / DELTA)

    target_up = 0.5 * (1.0 + p)
    target_dn = 0.5 * (1.0 - p)
    abs_gap_up = target_up * gap_up / 100.0
    abs_gap_dn = target_dn * gap_dn / 100.0
    eps_eff = float(np.exp(-8.0 * np.pi * shift))

    if history.shape[0] >= 6:
        late = history[-5:, 1]
        contact_drift = float((late[-1] - late[0]) / max(abs(late[-1]), 1.0e-14))
    else:
        contact_drift = float("nan")

    return {
        "run": os.path.basename(run_dir),
        "control": bool(control),
        "P": float(p),
        "iteration": int(it),
        "contact": float(contact),
        "shift": float(shift),
        "eps_eff": eps_eff,
        "contact_over_eps": float(contact / eps_eff),
        "gap_up_pct": float(gap_up),
        "gap_dn_pct": float(gap_dn),
        "abs_gap_up": float(abs_gap_up),
        "abs_gap_dn": float(abs_gap_dn),
        "qff": qff,
        "q_star": q_star,
        "q_star_minus_qff": float(q_star - qff),
        "max_shifted_delta": max_shifted_delta,
        "advantage_delta": advantage_delta,
        "contact_drift_last5": contact_drift,
        "q": q,
        "re0": re0,
    }


def discover(base):
    rows = []
    for run_dir in sorted(glob.glob(os.path.join(base, "P0p*_etaexact"))):
        m = TAG_RE.match(os.path.basename(run_dir))
        if m:
            rows.append(load_run(run_dir, int(m.group(1)) / 100.0))
    rows.sort(key=lambda x: x["P"])
    if not rows:
        raise RuntimeError(f"nessun P0p*_etaexact trovato in {base}")

    controls = []
    mdir = os.path.join(base, "M_p12_etaexact")
    if os.path.isdir(mdir):
        with open(os.path.join(mdir, "config.json")) as fh:
            cfg = json.load(fh)
        mu_up = float(cfg.get("_mu_up", 1.65))
        mu_dn = float(cfg.get("_mu_dn", 0.35))
        controls.append(load_run(mdir, (mu_up - mu_dn) / (mu_up + mu_dn), control=True))
    return rows, controls


def shade_transition(ax):
    ax.axvspan(0.65, 0.70, color="#eda100", alpha=0.10, lw=0)


def phase_figure(rows, controls, dst):
    p = np.array([r["P"] for r in rows])
    eps = np.array([r["eps_eff"] for r in rows])
    contact = np.array([r["contact"] for r in rows])
    qff = np.array([r["qff"] for r in rows])
    qs = np.array([r["q_star"] for r in rows])
    lam = np.array([r["max_shifted_delta"] for r in rows])
    gu = np.array([r["gap_up_pct"] for r in rows])
    gd = np.array([r["gap_dn_pct"] for r in rows])
    au = np.array([r["abs_gap_up"] for r in rows])
    ad = np.array([r["abs_gap_dn"] for r in rows])

    fig, ax = plt.subplots(2, 3, figsize=(13.5, 7.6), layout="constrained")
    for a in ax.flat:
        shade_transition(a)

    ax[0, 0].semilogy(p, eps, "o-", color=BLUE, label=r"$\epsilon_{0,\rm eff}$")
    ax[0, 0].axhline(1.0, color=MUTED, ls=":", lw=1)
    ax[0, 0].set(title="Linea di Thouless ripinnata", xlabel="P", ylabel=r"$\epsilon_{0,\rm eff}$")

    ax[0, 1].semilogy(p, contact, "o-", color=ORANGE, label="scan da seed libero")
    ax[0, 1].set(title="Contact della coppia", xlabel="P", ylabel="C")

    ax[0, 2].plot(p, qff, "--", color=MUTED, label=r"$q_{\rm FF}$ imposto")
    ax[0, 2].plot(p, qs, "o-", color=PURPLE, label=r"$Q_*$, massimo globale")
    ax[0, 2].set(title="Canale dominante a $\omega=0$", xlabel="P", ylabel="Q")
    ax[0, 2].legend(loc="upper left")

    stable = lam <= 0.0
    ax[1, 0].plot(p, lam, color=MUTED, lw=1)
    ax[1, 0].scatter(p[stable], lam[stable], color=GREEN, label="stabile")
    ax[1, 0].scatter(p[~stable], lam[~stable], color=RED, label="supercritico")
    ax[1, 0].axhline(0.0, color=INK, ls=":", lw=1)
    ax[1, 0].set(title="Massimo dopo il pinning a $q_{FF}$", xlabel="P",
                 ylabel=r"$\max_Q\,\mathrm{Re}\,\Gamma^{-1}_{shift}(Q,0)/\delta$")
    ax[1, 0].legend(loc="upper left")

    ax[1, 1].plot(p, gu, "o-", color=BLUE, label=r"$\uparrow$")
    ax[1, 1].plot(p, gd, "s-", color=ORANGE, label=r"$\downarrow$")
    ax[1, 1].axhline(0.0, color=MUTED, ls=":", lw=1)
    ax[1, 1].set_yscale("symlog", linthresh=1.0)
    ax[1, 1].set(title="Errore di densita relativo", xlabel="P", ylabel="gap [%]")
    ax[1, 1].legend(loc="upper left")

    ax[1, 2].plot(p, au, "o-", color=BLUE, label=r"$\Delta n_\uparrow$")
    ax[1, 2].plot(p, ad, "s-", color=ORANGE, label=r"$\Delta n_\downarrow$")
    ax[1, 2].axhline(0.0, color=MUTED, ls=":", lw=1)
    ax[1, 2].set(title="Errore di densita assoluto", xlabel="P", ylabel=r"$n-\mu_\sigma/2$")
    ax[1, 2].legend(loc="upper left")

    for c in controls:
        kw = dict(marker="*", s=115, color=INK, zorder=5, label="M, seed warm")
        ax[0, 0].scatter([c["P"]], [c["eps_eff"]], **kw)
        ax[0, 1].scatter([c["P"]], [c["contact"]], **kw)
        ax[0, 2].scatter([c["P"]], [c["q_star"]], **kw)
        ax[1, 0].scatter([c["P"]], [c["max_shifted_delta"]], **kw)
        ax[1, 1].scatter([c["P"]], [c["gap_dn_pct"]], **kw)
        ax[1, 2].scatter([c["P"]], [c["abs_gap_dn"]], **kw)
    ax[0, 1].legend(loc="upper left")

    fig.suptitle(
        "X_FFLO_SLIM — scan in polarizzazione sulla linea ripinnata\n"
        "fascia gialla: cambio del massimo da FFLO a Q=0 (non una frontiera termodinamica)",
        color=INK,
    )
    fig.savefig(dst, dpi=180)
    plt.close(fig)


def gamma_figure(rows, dst):
    cmap = plt.get_cmap("viridis")
    norm = Normalize(min(r["P"] for r in rows), max(r["P"] for r in rows))
    fig, ax = plt.subplots(1, 2, figsize=(12.2, 4.5), layout="constrained")
    for r in rows:
        q, re0 = r["q"], r["re0"]
        shifted = (re0 - r["shift"] - DELTA) / DELTA
        m = q <= 1.35
        col = cmap(norm(r["P"]))
        ax[0].plot(q[m], shifted[m], color=col, label=f"P={r['P']:.2f}")
        ax[1].plot(r["P"], r["advantage_delta"], "o", color=col)

    ax[0].axhline(0.0, color=INK, ls=":", lw=1)
    ax[0].set(title=r"Profilo dopo il pinning ($q_{FF}$ e a $-\delta$)", xlabel="Q",
              ylabel=r"$\mathrm{Re}\,\Gamma^{-1}_{shift}(Q,0)/\delta$", ylim=(-15, 10))
    ax[1].plot([r["P"] for r in rows], [r["advantage_delta"] for r in rows],
               color=MUTED, lw=1)
    ax[1].axhline(1.0, color=INK, ls=":", lw=1, label="supercritico dopo -delta")
    ax[1].axhline(0.0, color=MUTED, ls="--", lw=1, label=r"$q_{FF}$ e massimo globale")
    shade_transition(ax[1])
    ax[1].set(title=r"Vantaggio del massimo globale su $q_{FF}$", xlabel="P",
              ylabel=r"$[\max_Q\mathrm{Re}\Gamma^{-1}-\mathrm{Re}\Gamma^{-1}(q_{FF})]/\delta$")
    ax[1].legend(loc="upper left")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, label="P")
    fig.suptitle("Competizione fra il ramo FFLO e il canale Q=0", color=INK)
    fig.savefig(dst, dpi=180)
    plt.close(fig)


def write_csv(rows, controls, dst):
    fields = [
        "run", "control", "P", "iteration", "contact", "shift", "eps_eff",
        "contact_over_eps", "gap_up_pct", "gap_dn_pct", "abs_gap_up", "abs_gap_dn",
        "qff", "q_star", "q_star_minus_qff", "max_shifted_delta", "advantage_delta",
        "contact_drift_last5",
    ]
    with open(dst, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in rows + controls:
            w.writerow({k: row[k] for k in fields})


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    base = argv[0] if argv else os.path.join("out", "cluster", "P_scan")
    rows, controls = discover(base)
    out = os.path.join(base, "plots")
    os.makedirs(out, exist_ok=True)
    phase_figure(rows, controls, os.path.join(out, "phase_diagram.png"))
    gamma_figure(rows, os.path.join(out, "gamma_competition.png"))
    write_csv(rows, controls, os.path.join(out, "phase_diagram.csv"))

    print(" P    C        eps_eff   Q*       qff      max_shift/d   gap_up%   gap_dn%")
    for r in rows:
        print(f"{r['P']:4.2f}  {r['contact']:7.4f}  {r['eps_eff']:8.4f}  "
              f"{r['q_star']:7.4f}  {r['qff']:7.4f}  {r['max_shifted_delta']:+11.3f}  "
              f"{r['gap_up_pct']:+8.3f}  {r['gap_dn_pct']:+9.3f}")
    print("scritti in", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
