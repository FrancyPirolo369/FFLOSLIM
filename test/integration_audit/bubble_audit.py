#!/usr/bin/env python3
"""Audit degli integrali della bolla: pairbuild di produzione (ricetta FINAL) su un sottoinsieme di Q,
una manopola alla volta.

Stato: RUN, iterazione IT; cube d'ingresso = out/cluster/RUN/iter{IT-1}/next_cubes (quelle di density
in quella iterazione).  pairbuild con gli stessi argomenti di test/run_test.py per la ricetta FINAL
(turbo, Lambda 4, reticolo dw 2.5e-4 coda 100, finestre in p 0.08 x 21 kf_only, 21 nodi angolari
attorno a kF, residuo nella sola finestra T = 0, omega feature zero_and_thresholds 1.05 x 21), ma con
--q-points = righe scelte della griglia Q di produzione (Q0+, frazioni di qff, 2, 4, 8; pairbuild
aggiunge 0, qff e q_table_max).  Le feature in omega nascono dalle Q della lista, quindi la griglia in
Omega e' un po' piu' rada di quella di produzione: ImG^-1 e' puntuale (confronto diretto con la snap),
la KK di ReG^-1 no (piccola differenza, si confronta ogni variante con la propria base).

Varianti:
    pnodes61   --p-nodes 61                    griglia uniforme in p 31 -> 61 (finestre a kF invariate)
    pwin       --p-feature-width 0.16 --p-feature-nodes 41   finestre in p piu' larghe e fitte
    ang20      --angular-nodes 20              pannelli angolari 10 -> 20
    angfeat41  --angular-feature-nodes 41      nodi angolari attorno a kF del partner 21 -> 41
    dw125      --lattice-dw 1.25e-4            reticolo eps/Omega 2.5e-4 -> 1.25e-4
    tail200    --lattice-n-tail 200            coda del reticolo 100 -> 200
    coh2       --coherent-nk 192 --coherent-nquad 96   parte coerente QPxQP analitica, griglie x2
Lettura: Delta ReG^-1(qff, 0) -> Delta g_c a un passo = -4 pi Delta; gara Q0+ contro qff; ImG^-1(qff, Omega)
vicino a 0; scarto massimo di ReG^-1(Q, 0) sulle righe.

Uso (dalla radice di SLIM, con un tetto di memoria):
    systemd-run --user --scope -p MemoryMax=5G python3 test/integration_audit/bubble_audit.py compute
    python3 test/integration_audit/bubble_audit.py report
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DELTA = 1.0e-3
FRACS = (0.25, 0.5, 0.9, 1.04, 1.1, 1.5, 2.0, 3.0)
ABS_Q = (2.0, 4.0, 8.0)
FINAL = ["--profile", "turbo", "--lattice-n-tail", "100", "--angular-feature-nodes", "21",
         "--p-feature-width", "0.08", "--p-feature-nodes", "21", "--p-feature-mode", "kf_only",
         "--eps-window-only", "--lattice-dw", "0.00025",
         "--omega-feature-mode", "zero_and_thresholds", "--omega-feature-nodes", "21"]
VARIANTS = {
    "base": [],
    "pnodes61": ["--p-nodes", "61"],
    "pwin": ["--p-feature-width", "0.16", "--p-feature-nodes", "41"],
    "ang20": ["--angular-nodes", "20"],
    "angfeat41": ["--angular-feature-nodes", "41"],
    "dw125": ["--lattice-dw", "0.000125"],
    "tail200": ["--lattice-n-tail", "200"],
    "coh2": ["--coherent-nk", "192", "--coherent-nquad", "96"],
    # control variate esatto (2026-10-02): A0 analitico nei punti esatti, pavimento storico 1e-3
    "ctrlan": ["--control-analytic"],
    # ... e pavimento di larghezza piu' basso, uguale per A0 e per la parte analitica
    "ctrlan_f3e4": ["--control-analytic", "--qp-gamma-floor", "3e-4"],
    "ctrlan_f1e4": ["--control-analytic", "--qp-gamma-floor", "1e-4"],
    # griglie graduate verso kF (2026-10-02): variabili d'ambiente, vedi VARIANT_ENV
    "grad_p": [],
    "grad_phi": [],
    "grad": [],
    "grad5e5": [],
    # griglie graduate anche nella parte coerente analitica (angolo a pannelli + k graduato)
    "gradcoh": ["--coherent-angle-mode", "phipanel"],
    # versioni a costo ~ produzione: nodi spostati verso kF invece di aggiunti
    "base2": [],          # base ricronometrata nelle stesse condizioni delle varianti lean
    "pauli5e3": [],       # blocco Omega uniforme sui bordi di Pauli (OMEGA_PAULI_STEP)
    "pauli1e2": [],
    "leanA": ["--coherent-angle-mode", "phipanel", "--p-feature-nodes", "12"],
    "leanB": ["--coherent-angle-mode", "phipanel", "--p-feature-nodes", "12", "--angular-feature-nodes", "12"],
    # attribuzione FINAL -> FINAL2 a un passo (2026-10-03): base = bolla di FINAL
    "f2": ["--coherent-angle-mode", "phipanel", "--p-feature-nodes", "16"],          # bolla di FINAL2
    # convergenza in Lambda della bolla numerica (2026-10-05): stessi nodi per unita' di p
    "f2L6": ["--coherent-angle-mode", "phipanel", "--p-feature-nodes", "16", "--bubble-p-int-max", "6", "--p-nodes", "46"],
    # A0 del controllo esatto dal modello QP invece che interpolato fra le righe della cube (artefatto al bordo, 2026-10-05)
    "f2L8c384": ["--coherent-angle-mode", "phipanel", "--p-feature-nodes", "16", "--bubble-p-int-max", "8", "--p-nodes", "62",
            "--control-analytic", "--coherent-nk", "384"],
    "f2L8p100c384": ["--coherent-angle-mode", "phipanel", "--p-feature-nodes", "16", "--bubble-p-int-max", "8", "--p-nodes", "100",
            "--control-analytic", "--coherent-nk", "384"],
    "f2L8p200c384": ["--coherent-angle-mode", "phipanel", "--p-feature-nodes", "16", "--bubble-p-int-max", "8", "--p-nodes", "200",
            "--control-analytic", "--coherent-nk", "384"],
    "f2p62c192": ["--coherent-angle-mode", "phipanel", "--p-feature-nodes", "16", "--bubble-p-int-max", "4", "--p-nodes", "62",
            "--control-analytic", "--coherent-nk", "192"],
    "f2p31c192": ["--coherent-angle-mode", "phipanel", "--p-feature-nodes", "16", "--bubble-p-int-max", "4", "--p-nodes", "31",
            "--control-analytic", "--coherent-nk", "192"],
    "f2L8p100k2": ["--coherent-angle-mode", "phipanel", "--p-feature-nodes", "16", "--bubble-p-int-max", "8", "--p-nodes", "100",
            "--control-analytic", "--coherent-nk", "384"],
    "f2L8p140k2": ["--coherent-angle-mode", "phipanel", "--p-feature-nodes", "16", "--bubble-p-int-max", "8", "--p-nodes", "140",
            "--control-analytic", "--coherent-nk", "384"],
    "f2p31k2": ["--coherent-angle-mode", "phipanel", "--p-feature-nodes", "16", "--bubble-p-int-max", "4", "--p-nodes", "31",
            "--control-analytic", "--coherent-nk", "192"],
    "f2L8p140": ["--coherent-angle-mode", "phipanel", "--p-feature-nodes", "16", "--bubble-p-int-max", "8", "--p-nodes", "140",
                 "--control-analytic", "--coherent-nk", "384"],
    "f2L8cap": ["--coherent-angle-mode", "phipanel", "--p-feature-nodes", "16", "--bubble-p-int-max", "8", "--p-nodes", "100",
                "--control-analytic"],
    "f2L8ca": ["--coherent-angle-mode", "phipanel", "--p-feature-nodes", "16", "--bubble-p-int-max", "8", "--p-nodes", "62",
               "--control-analytic"],
    "f2L6ca": ["--coherent-angle-mode", "phipanel", "--p-feature-nodes", "16", "--bubble-p-int-max", "6", "--p-nodes", "46",
               "--control-analytic"],
    "f2ca": ["--coherent-angle-mode", "phipanel", "--p-feature-nodes", "16", "--control-analytic"],
    # parte coerente A0*A0 senza il taglio Lambda (fino a k = 20): residuo che si annulla da solo, niente taper
    "f2cac20": ["--coherent-angle-mode", "phipanel", "--p-feature-nodes", "16", "--control-analytic",
                "--coherent-kmax", "20"],
    "f2L6cac20": ["--coherent-angle-mode", "phipanel", "--p-feature-nodes", "16", "--bubble-p-int-max", "6",
                  "--p-nodes", "46", "--control-analytic", "--coherent-kmax", "19"],
    "f2L8": ["--coherent-angle-mode", "phipanel", "--p-feature-nodes", "16", "--bubble-p-int-max", "8", "--p-nodes", "62"],
    "f2_pgrad": ["--p-feature-nodes", "16"],                                          # solo finestre p graduate
    "f2_coh": ["--coherent-angle-mode", "phipanel"],                                  # solo parte coerente lean
    "unif41": ["--p-feature-nodes", "41"],                                            # FINAL con finestre x2
    # riferimento fitto: finestre p graduate 81 nodi su +-0.16, griglia p 61, parte coerente graduata piena
    "ref": ["--coherent-angle-mode", "phipanel", "--p-nodes", "61", "--p-feature-width", "0.16",
            "--p-feature-nodes", "81", "--angular-feature-nodes", "41"],
}
VARIANT_ENV = {
    "pauli5e3": {"OMEGA_PAULI_STEP": "5e-3"},
    "pauli1e2": {"OMEGA_PAULI_STEP": "1e-2"},
    "grad_p": {"IMPI_P_GRADED_MIN": "1e-4"},
    "grad_phi": {"PHIPANEL_GRADED_MIN": "1e-4"},
    "grad": {"IMPI_P_GRADED_MIN": "1e-4", "PHIPANEL_GRADED_MIN": "1e-4"},
    "grad5e5": {"IMPI_P_GRADED_MIN": "5e-5", "PHIPANEL_GRADED_MIN": "5e-5"},
    "gradcoh": {"IMPI_P_GRADED_MIN": "1e-4", "PHIPANEL_GRADED_MIN": "1e-4", "COHERENT_K_GRADED_MIN": "1e-4",
                "COHERENT_PHI_GRADED_MIN": "1e-4"},
    "leanA": {"IMPI_P_GRADED_MIN": "1e-4", "COHERENT_PHI_GRADED_MIN": "1e-4", "COHERENT_PHI_PANELS": "12",
              "COHERENT_PHI_FEATURE_N": "12", "COHERENT_K_GRADED_MIN": "1e-4", "COHERENT_K_GRADED_N": "40"},
    "leanB": {"IMPI_P_GRADED_MIN": "1e-4", "PHIPANEL_GRADED_MIN": "1e-4", "COHERENT_PHI_GRADED_MIN": "1e-4",
              "COHERENT_PHI_PANELS": "12", "COHERENT_PHI_FEATURE_N": "12", "COHERENT_K_GRADED_MIN": "1e-4",
              "COHERENT_K_GRADED_N": "40"},
    # come in test/run_test.py con le manopole di submit_final2.sh
    "f2": {"IMPI_P_GRADED_MIN": "1e-4", "COHERENT_K_GRADED_MIN": "1e-4", "COHERENT_PHI_GRADED_MIN": "1e-4",
           "COHERENT_K_GRADED_N": "40", "COHERENT_PHI_PANELS": "12", "COHERENT_PHI_FEATURE_N": "12"},
    "f2_pgrad": {"IMPI_P_GRADED_MIN": "1e-4"},
    "f2L6": {"IMPI_P_GRADED_MIN": "1e-4", "COHERENT_K_GRADED_MIN": "1e-4", "COHERENT_PHI_GRADED_MIN": "1e-4",
             "COHERENT_K_GRADED_N": "40", "COHERENT_PHI_PANELS": "12", "COHERENT_PHI_FEATURE_N": "12"},
    "f2L8c384": {"IMPI_P_GRADED_MIN": "1e-4", "COHERENT_K_GRADED_MIN": "1e-4", "COHERENT_PHI_GRADED_MIN": "1e-4", "COHERENT_K_GRADED_N": "40", "COHERENT_PHI_PANELS": "12", "COHERENT_PHI_FEATURE_N": "12"},
    "f2L8p100c384": {"IMPI_P_GRADED_MIN": "1e-4", "COHERENT_K_GRADED_MIN": "1e-4", "COHERENT_PHI_GRADED_MIN": "1e-4", "COHERENT_K_GRADED_N": "40", "COHERENT_PHI_PANELS": "12", "COHERENT_PHI_FEATURE_N": "12"},
    "f2L8p200c384": {"IMPI_P_GRADED_MIN": "1e-4", "COHERENT_K_GRADED_MIN": "1e-4", "COHERENT_PHI_GRADED_MIN": "1e-4", "COHERENT_K_GRADED_N": "40", "COHERENT_PHI_PANELS": "12", "COHERENT_PHI_FEATURE_N": "12"},
    "f2p62c192": {"IMPI_P_GRADED_MIN": "1e-4", "COHERENT_K_GRADED_MIN": "1e-4", "COHERENT_PHI_GRADED_MIN": "1e-4", "COHERENT_K_GRADED_N": "40", "COHERENT_PHI_PANELS": "12", "COHERENT_PHI_FEATURE_N": "12"},
    "f2p31c192": {"IMPI_P_GRADED_MIN": "1e-4", "COHERENT_K_GRADED_MIN": "1e-4", "COHERENT_PHI_GRADED_MIN": "1e-4", "COHERENT_K_GRADED_N": "40", "COHERENT_PHI_PANELS": "12", "COHERENT_PHI_FEATURE_N": "12"},
    "f2L8p100k2": {"IMPI_P_GRADED_MIN": "1e-4", "COHERENT_K_GRADED_MIN": "1e-4", "COHERENT_PHI_GRADED_MIN": "1e-4", "COHERENT_K_GRADED_N": "40", "COHERENT_PHI_PANELS": "12", "COHERENT_PHI_FEATURE_N": "12", "QP_E_INTERP_K2": "1"},
    "f2L8p140k2": {"IMPI_P_GRADED_MIN": "1e-4", "COHERENT_K_GRADED_MIN": "1e-4", "COHERENT_PHI_GRADED_MIN": "1e-4", "COHERENT_K_GRADED_N": "40", "COHERENT_PHI_PANELS": "12", "COHERENT_PHI_FEATURE_N": "12", "QP_E_INTERP_K2": "1"},
    "f2p31k2": {"IMPI_P_GRADED_MIN": "1e-4", "COHERENT_K_GRADED_MIN": "1e-4", "COHERENT_PHI_GRADED_MIN": "1e-4", "COHERENT_K_GRADED_N": "40", "COHERENT_PHI_PANELS": "12", "COHERENT_PHI_FEATURE_N": "12", "QP_E_INTERP_K2": "1"},
    "f2L8p140": {"IMPI_P_GRADED_MIN": "1e-4", "COHERENT_K_GRADED_MIN": "1e-4", "COHERENT_PHI_GRADED_MIN": "1e-4",
                 "COHERENT_K_GRADED_N": "40", "COHERENT_PHI_PANELS": "12", "COHERENT_PHI_FEATURE_N": "12"},
    "f2L8cap": {"IMPI_P_GRADED_MIN": "1e-4", "COHERENT_K_GRADED_MIN": "1e-4", "COHERENT_PHI_GRADED_MIN": "1e-4",
                "COHERENT_K_GRADED_N": "40", "COHERENT_PHI_PANELS": "12", "COHERENT_PHI_FEATURE_N": "12"},
    "f2L8ca": {"IMPI_P_GRADED_MIN": "1e-4", "COHERENT_K_GRADED_MIN": "1e-4", "COHERENT_PHI_GRADED_MIN": "1e-4",
               "COHERENT_K_GRADED_N": "40", "COHERENT_PHI_PANELS": "12", "COHERENT_PHI_FEATURE_N": "12"},
    "f2L6ca": {"IMPI_P_GRADED_MIN": "1e-4", "COHERENT_K_GRADED_MIN": "1e-4", "COHERENT_PHI_GRADED_MIN": "1e-4",
               "COHERENT_K_GRADED_N": "40", "COHERENT_PHI_PANELS": "12", "COHERENT_PHI_FEATURE_N": "12"},
    "f2cac20": {"IMPI_P_GRADED_MIN": "1e-4", "COHERENT_K_GRADED_MIN": "1e-4", "COHERENT_PHI_GRADED_MIN": "1e-4",
                "COHERENT_K_GRADED_N": "40", "COHERENT_PHI_PANELS": "12", "COHERENT_PHI_FEATURE_N": "12"},
    "f2L6cac20": {"IMPI_P_GRADED_MIN": "1e-4", "COHERENT_K_GRADED_MIN": "1e-4", "COHERENT_PHI_GRADED_MIN": "1e-4",
                  "COHERENT_K_GRADED_N": "40", "COHERENT_PHI_PANELS": "12", "COHERENT_PHI_FEATURE_N": "12"},
    "f2ca": {"IMPI_P_GRADED_MIN": "1e-4", "COHERENT_K_GRADED_MIN": "1e-4", "COHERENT_PHI_GRADED_MIN": "1e-4",
             "COHERENT_K_GRADED_N": "40", "COHERENT_PHI_PANELS": "12", "COHERENT_PHI_FEATURE_N": "12"},
    "f2L8": {"IMPI_P_GRADED_MIN": "1e-4", "COHERENT_K_GRADED_MIN": "1e-4", "COHERENT_PHI_GRADED_MIN": "1e-4",
             "COHERENT_K_GRADED_N": "40", "COHERENT_PHI_PANELS": "12", "COHERENT_PHI_FEATURE_N": "12"},
    "f2_coh": {"COHERENT_K_GRADED_MIN": "1e-4", "COHERENT_PHI_GRADED_MIN": "1e-4", "COHERENT_K_GRADED_N": "40",
               "COHERENT_PHI_PANELS": "12", "COHERENT_PHI_FEATURE_N": "12"},
    "ref": {"IMPI_P_GRADED_MIN": "1e-4", "PHIPANEL_GRADED_MIN": "1e-4", "COHERENT_K_GRADED_MIN": "1e-4",
            "COHERENT_PHI_GRADED_MIN": "1e-4", "COHERENT_K_GRADED_N": "120", "COHERENT_PHI_PANELS": "24",
            "COHERENT_PHI_FEATURE_N": "21"},
}


def override(base, extra):
    """argomenti di base con quelli della variante al posto degli omonimi (anche flag senza valore)"""
    out = list(base)
    i = 0
    while i < len(extra):
        key = extra[i]
        has_val = i + 1 < len(extra) and not extra[i + 1].startswith("--")
        if has_val:
            if key in out:
                out[out.index(key) + 1] = extra[i + 1]
            else:
                out += [key, extra[i + 1]]
            i += 2
        else:
            if key not in out:
                out.append(key)
            i += 1
    return out


def work_dir(run, it, qset="audit"):
    suffix = "" if qset == "audit" else f"_{qset}"
    return os.path.join(HERE, "out", "integration_audit", f"bubble_{run}_it{it}{suffix}")


def q_rows(q, qff, qset):
    """righe Q: 'audit' = sottoinsieme di quelle di produzione; 'mid' = righe di produzione in
    [0.8, 1.3] qff piu' i punti medi (per misurare l'errore dell'interpolazione in Q di Sigma)"""
    if qset == "race":
        # gara fra canali ad alta P: righe di produzione fino a 0.4 qff con i punti medi, e attorno a qff
        low = q[q <= 0.4 * qff]
        near = q[(q >= 0.9 * qff) & (q <= 1.1 * qff)]
        return np.unique(np.concatenate([low, 0.5 * (low[:-1] + low[1:]), near]))
    if qset == "edge":
        # solo la riga Q0+ (pairbuild aggiunge 0, qff e la riga UV): prova veloce del bordo di Lambda
        nz = np.flatnonzero(q > 1e-9)
        return np.array([q[nz[np.argmin(q[nz])]]])
    if qset == "large":
        # righe di produzione a Q grande (Lambda < Q < 2 Lambda e oltre), dove la bolla con |p| <= Lambda
        # perde parte della shell: test di convergenza in Lambda (2026-10-05)
        nz = np.flatnonzero(q > 1e-9)
        big = q[(q >= 2.0) & (q <= 9.0)]
        return np.unique(np.concatenate([[q[nz[np.argmin(q[nz])]]], big]))
    if qset == "small":
        # righe veloci per provare il residuo vicino a Omega = 0
        nz = np.flatnonzero(q > 1e-9)
        pick = {int(nz[np.argmin(q[nz])])}
        pick |= {int(np.argmin(np.abs(q - f * qff))) for f in (0.5, 0.9, 1.04, 1.1, 1.5)}
        return np.sort(q[sorted(pick)])
    if qset == "dense":
        # tutte le righe di produzione + i punti medi fino a 4 qff (stessa corsa, stessa griglia Omega)
        core = q[q <= 4.0 * qff]
        return np.unique(np.concatenate([q, 0.5 * (core[:-1] + core[1:])]))
    if qset in ("mid", "mid2"):
        if qset == "mid":
            core = q[(q >= 0.8 * qff) & (q <= 1.3 * qff)]
            return np.unique(np.concatenate([core, 0.5 * (core[:-1] + core[1:])]))
        out = []
        for lo, hi in ((0.05, 0.8), (1.3, 3.0)):          # 'mid2': fuori dalla zona critica
            core = q[(q >= lo * qff) & (q <= hi * qff)]
            out += [core, 0.5 * (core[:-1] + core[1:])]
        return np.unique(np.concatenate(out))
    nz = np.flatnonzero(q > 1e-9)
    pick = {int(nz[np.argmin(q[nz])])}
    pick |= {int(np.argmin(np.abs(q - f * qff))) for f in FRACS}
    pick |= {int(np.argmin(np.abs(q - v))) for v in ABS_Q}
    return np.sort(q[sorted(pick)])


def state(run, it, base, cubes=None):
    rd = os.path.join(base, run)
    if cubes:
        # cube esplicite (es. seeds_warm/P0pXX_final2 = ingresso dell'iterazione 1 di FINAL2)
        up = glob.glob(os.path.join(cubes, "A_komega_spinup*.npz"))[0]
        dn = glob.glob(os.path.join(cubes, "A_komega_spindown*.npz"))[0]
    else:
        cdir = os.path.join(rd, f"iter{it - 1:03d}", "next_cubes")
        up = glob.glob(os.path.join(cdir, "A_komega_spinup_iter*.npz"))[0]
        dn = glob.glob(os.path.join(cdir, "A_komega_spindown_iter*.npz"))[0]
    z = np.load(os.path.join(rd, "snap", f"iter{it:03d}.npz"))
    with np.load(up) as c:
        mu_up, mu_dn = float(c["mu_up"]), float(c["mu_dn"])
        kmax = float(np.max(c["k"]))
    return up, dn, z, mu_up, mu_dn, kmax


def compute(a):
    up, dn, z, mu_up, mu_dn, kmax = state(a.run, a.it, a.base, getattr(a, 'cubes', None))
    q = np.asarray(z["q"], float)
    qff = float(z["qff"])
    qs = q_rows(q, qff, a.qset)
    width = abs(mu_up - mu_dn) + 0.05
    wd = work_dir(a.run, a.it, a.qset)
    os.makedirs(wd, exist_ok=True)
    for name in a.variants.split(","):
        out = os.path.join(wd, name)
        if os.path.exists(os.path.join(out, "pair", "pair_gamma_table.npz")):
            print(f"[{name}] gia' fatto, salto", flush=True)
            continue
        args = override(FINAL + ["--omega-feature-half-width", f"{width:.15g}"], VARIANTS[name])
        # come in produzione: la riga UV sta a k_max - Lambda, altrimenti adn_phi taglia Lambda a k_max - Q_max
        lam = float(args[args.index("--bubble-p-int-max") + 1]) if "--bubble-p-int-max" in args else 4.0
        q_table_max = min(20.0, kmax - lam - 0.1)
        cmd = [sys.executable, "-m", "fflo.pairbuild", "--up", up, "--down", dn, "--out-dir", out,
               "--workers", str(a.workers), "--q-points", ",".join(f"{v:.15g}" for v in qs),
               "--q-table-max", f"{q_table_max:.15g}"] + args
        env = dict(os.environ)
        for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[v] = "1"
        env.update(VARIANT_ENV.get(name, {}))
        t0 = time.time()
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "pairbuild.log"), "w") as fh:
            rc = subprocess.run(cmd, env=env, cwd=HERE, stdout=fh, stderr=subprocess.STDOUT).returncode
        print(f"[{name}] rc={rc} in {time.time() - t0:.0f} s  ({qs.size} righe Q + 0, qff, {q_table_max:g})",
              flush=True)
        if rc:
            return rc
    return 0


def load_pair(path):
    with np.load(path, allow_pickle=True) as t:
        return (np.asarray(t["q"], float), np.asarray(t["omega"], float),
                np.asarray(t["ReInvGamma"], float), np.asarray(t["ImInvGamma"], float))


def report(a):
    wd = work_dir(a.run, a.it)
    _, _, z, _, _, _ = state(a.run, a.it, a.base, getattr(a, 'cubes', None))
    qff = float(z["qff"])
    tabs = {}
    for name in VARIANTS:
        p = os.path.join(wd, name, "pair", "pair_gamma_table.npz")
        if os.path.exists(p):
            tabs[name] = load_pair(p)
    if "base" not in tabs:
        print("manca la base")
        return 1
    qb, wb, reb, imb = tabs["base"]
    i0 = int(np.argmin(np.abs(wb)))
    iq = int(np.argmin(np.abs(qb - qff)))
    nz = np.flatnonzero(qb > 1e-9)
    i0q = int(nz[np.argmin(qb[nz])])
    # convalida: ImG^-1 puntuale contro la snap, sulle (Q, Omega) comuni
    qs, ws = np.asarray(z["q"], float), np.asarray(z["omega"], float)
    ims, res = np.asarray(z["ImInvGamma"], float), np.asarray(z["ReInvGamma"], float)
    print("convalida base contro la snap di produzione (stessa iterazione):")
    for i, qv in enumerate(qb):
        j = int(np.argmin(np.abs(qs - qv)))
        if abs(qs[j] - qv) > 1e-9:
            continue
        common, ib, js = np.intersect1d(np.round(wb, 12), np.round(ws, 12), return_indices=True)
        m = np.abs(common) < 2.0
        d_im = np.max(np.abs(imb[i, ib][m] - ims[j, js][m])) / max(np.max(np.abs(ims[j, js][m])), 1e-30)
        print(f"   Q/qff = {qv / qff:7.3f}: max|dImG^-1|/max|ImG^-1| (|w|<2) = {d_im:.1e}   "
              f"ReG^-1(Q,0): base {reb[i, i0]:+.6f}  snap {res[j, int(np.argmin(np.abs(ws)))]:+.6f}")
    print(f"\n{'variante':10s} {'dReG^-1(qff,0)':>15s} {'-> dg_c':>8s} {'d(gara Q0+-qff)':>16s} "
          f"{'max|dReG^-1(Q,0)|':>18s} {'riga':>6s}   dImG^-1(qff,w)/ImG^-1 a w = -0.01 -0.002 +0.002 +0.01 +0.05")
    for name, (q, w, re, im) in tabs.items():
        if not np.allclose(q, qb):
            print(f"{name:10s} griglia Q diversa, salto")
            continue
        re0 = np.array([np.interp(0.0, w, r) for r in re])
        re0b = np.array([np.interp(0.0, wb, r) for r in reb])
        d = re0 - re0b
        dq = d[iq]
        race = (re0[i0q] - re0[iq]) - (re0b[i0q] - re0b[iq])
        j = int(np.argmax(np.abs(d)))
        rel = []
        for W in (-0.01, -0.002, 0.002, 0.01, 0.05):
            vb = np.interp(W, wb, imb[iq])
            v = np.interp(W, w, im[iq])
            rel.append((v - vb) / vb if vb != 0 else np.nan)
        print(f"{name:10s} {dq:+15.2e} {-4 * math.pi * dq:+8.4f} {race:+16.2e} {np.max(np.abs(d)):18.2e} "
              f"{qb[j] / qff:6.2f}   " + " ".join(f"{x:+.1e}" for x in rel))
    return 0


def qinterp(a):
    """A_pair calcolata nei punti medi contro l'interpolazione dalle righe vicine:
    A lineare in Q (produzione, sigma_engine panel_gl) oppure Gamma^-1 lineare in Q poi A."""
    _, _, z, _, _, _ = state(a.run, a.it, a.base, getattr(a, 'cubes', None))
    qff = float(z["qff"])
    q, w, re, im = load_pair(os.path.join(work_dir(a.run, a.it, a.qset if a.qset != "audit" else "mid"),
                                          "base", "pair", "pair_gamma_table.npz"))
    i0, iq = int(np.argmin(np.abs(w))), int(np.argmin(np.abs(q - qff)))
    re = re - re[iq, i0] - DELTA                       # pin come apply_shift_to_pair_table
    A = im / np.pi / (re ** 2 + im ** 2)               # A_pair = -Im(Gamma)/pi = Im(G^-1)/pi/|G^-1|^2
    zq = np.asarray(z["q"], float)
    prod = np.array([np.min(np.abs(zq - v)) < 1e-9 for v in q])
    bands = ((0.0, 0.01), (0.01, 0.05), (0.05, 0.5), (0.5, 5.0))
    print("A_pair nei punti medi fra righe di produzione: scarto relativo dell'interpolazione, int dOmega su bande di |Omega|")
    print(f"{'Q/qff':>7s}  {'interp.':>9s}  " + "  ".join(f"{lo:g}-{hi:g}".rjust(16) for lo, hi in bands))
    tot = {"A lin": np.zeros(len(bands)), "G^-1 lin": np.zeros(len(bands)), "ref": np.zeros(len(bands))}
    for i in range(1, q.size - 1):
        if prod[i] or not (prod[i - 1] and prod[i + 1]):
            continue
        t_ = (q[i] - q[i - 1]) / (q[i + 1] - q[i - 1])
        a_lin = (1 - t_) * A[i - 1] + t_ * A[i + 1]
        r_l = (1 - t_) * re[i - 1] + t_ * re[i + 1]
        i_l = (1 - t_) * im[i - 1] + t_ * im[i + 1]
        a_inv = i_l / np.pi / (r_l ** 2 + i_l ** 2)
        for lab, x in (("A lin", a_lin), ("G^-1 lin", a_inv)):
            cells = []
            for b, (lo, hi) in enumerate(bands):
                m = (np.abs(w) >= lo) & (np.abs(w) < hi)
                ref = np.trapezoid(np.abs(A[i][m]), w[m])
                d = np.trapezoid(x[m] - A[i][m], w[m])
                tot[lab][b] += q[i] * d
                if lab == "A lin":
                    tot["ref"][b] += q[i] * ref
                cells.append(f"{d / ref:+.2e}" if ref > 0 else "nan")
            print(f"{q[i] / qff:7.3f}  {lab:>9s}  " + "  ".join(c.rjust(16) for c in cells))
    print("somma pesata Q dQ sulle righe di mezzo (segno: + = l'interpolazione sovrastima il peso di coppia):")
    for lab in ("A lin", "G^-1 lin"):
        print(f"   {lab:9s} " + "  ".join(f"{tot[lab][b] / tot['ref'][b]:+.2e}".rjust(16) for b in range(len(bands))))
    return 0


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("compute", "report", "qinterp"))
    ap.add_argument("--run", default="P0p50_final")
    ap.add_argument("--it", type=int, default=8)
    ap.add_argument("--base", default=os.path.join(HERE, "out", "cluster"))
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cubes", default=None,
                    help="cartella con A_komega_spinup*.npz / spindown*.npz al posto di RUN/iter{IT-1}/next_cubes")
    ap.add_argument("--qset", choices=("audit", "mid", "mid2", "dense", "small", "race", "large", "edge"), default="audit")
    a = ap.parse_args(argv)
    if a.mode == "qinterp":
        return qinterp(a)
    return compute(a) if a.mode == "compute" else report(a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
