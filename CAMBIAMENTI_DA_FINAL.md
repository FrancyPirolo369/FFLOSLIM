# Cambiamenti da FINAL in poi (stato al 2026-10-06)

Riferimenti:
- ultimo commit: `55c23d4` del 2026-10-01 09:17 ("mappa: tre riferimenti file:riga corretti");
- FINAL lanciata il 2026-10-02;
- tutto quello che segue è codice NON committato, sincronizzato a mano sul cluster.

Regola generale: ogni modifica dopo FINAL è un'opzione nuova (flag o variabile d'ambiente) con il default uguale al comportamento precedente. Fanno eccezione i due punti segnati con ⚠ nella sezione 2.

## 1. Ricette (argomenti di run_test) e catena delle famiglie

**FINAL** (`test/submit_final.sh`, tag `P0pXX_final`, 2026-10-02):
```
--pintmax 12 --k-update-max 8 --eta-floor exact_zero --lattice-n-tail 100 --high-k-sigma pair-contact
--contact-tail --prune-keep 2 --thouless-q-mode qff-or-zero --p-feature-width 0.08 --p-feature-nodes 21
--p-feature-mode kf_only --lattice-dw 2.5e-4 --angular-feature-nodes 21 --sigma-nomega 193
--sigma-omega-dense-w 0.06 --sigma-omega-dense-stride 2 --sigma-nk 20 --sigma-k-kf-offsets 0.03:0.1:0.3:0.6
```
Default di run_test allora:
- `--impi-eps-window`, `--omega-chunk 1000`;
- taper `qkin_cosine` (stop 20, mobile);
- parte coerente `kjac`;
- alpha 0.3;
- nessun integrale fine del polo.

**FINAL2** (`test/submit_final2.sh`, default `RECIPE=final2`, tag `P0pXX_final2`, 2026-10-03) = FINAL con questi cambi:

| argomento | nuovo valore |
|---|---|
| `--p-feature-nodes` | 16 (era 21) |
| `--impi-p-graded-min` | 1e-4 (finestre p graduate) |
| `--coherent-angle-mode` | `phipanel` |
| `--coherent-graded-min` | 1e-4 |
| `--coherent-k-graded-n` | 40 |
| `--coherent-phi-panels` | 12 |
| `--coherent-phi-feature-n` | 12 |
| `--sigma-nk` | 18 (era 20) |
| `--sigma-k-kf-offsets` | `0.01:0.03:0.1:0.3:0.6` (aggiunto 0.01) |
| `--density-fine-pole` | on, è il nuovo default; solo diagnostica |

Pin: invariato (`qff-or-zero`).

**t_pauli** (2026-10-03) = FINAL2 + `--omega-pauli-step 1e-2`, `PIN=qff-or-zero-fit`. Superata.

**t_frz** (2026-10-04) = FINAL2 + `--taper-qfreeze 1 --ref-kk-fine-step 2.5e-4`, `PIN=qff-or-zero-fit`.

**t_qcut** (job 271825-28): codice di qcut con due bug. **Da scartare.**

**t_qcut2** = t_frz + `--taper-mode qcut_cosine --taper-stop 10` (codice corretto).

**HP** (`P0pXX_hp`, notte 2026-10-04) = t_qcut2 + `--uv-guard-lambda 2 --alpha 0.2`.

**hpL6** = HP + `--bubble-lambda 6 --p-nodes 46`. Instabile.

Partenze (seed copiati in `seeds_warm/<tag>` sul cluster):

| famiglia | parte da |
|---|---|
| FINAL | prod/gmax/x75a0p3, ultima iterazione completa |
| FINAL2 | FINAL, ultima iterazione |
| t_frz | FINAL2 |
| t_qcut2 | t_frz it12 |
| HP | t_qcut2 it8 (0.70, 0.85, 0.90); t_frz it12 (0.75, 0.95); t_frz it20 (0.80) |

## 2. Codice: diff rispetto a `55c23d4`, file per file

`+` = opzione nuova, spenta di default. `=` = rifattorizzazione con default identico. ⚠ = cambia il default.

- **`fflo/sigma_engine.py`**
  - `=` la griglia eps della tabella del fermione legge `SIGMA_PN_EPS_CORE_DW` (default 2e-3, identico; prima era fisso).
  - `=` velocizzazione riga per riga.
- **`fflo/qp_cube.py`**
  - `=` `build_qp_cube(..., gamma_floor=None)` salva anche il modello QP (`qp_model_*`).
  - `=` `_qp_rows` ha il pavimento opzionale; con None resta η, come prima.
- **`fflo/adn_phi.py`**
  - `+` pannelli angolari graduati (`PHIPANEL_GRADED_MIN`, default 0).
  - `+` `qp_model` per A0 analitico.
  - `=` resto invariato.
- **`fflo/analytic_bubble.py`**
  - `+` grappoli in k graduati (`COHERENT_K_GRADED_MIN/N`, default 0).
  - `+` angolo graduato in phipanel (`COHERENT_PHI_GRADED_MIN`).
- **`fflo/impi_table.py`**
  - `+` `--eps-window-only`: già usato da FINAL.
  - `+` `--control-analytic`.
  - `+` blocco Ω di Pauli (`OMEGA_PAULI_STEP`, default 0).
  - `+` finestre p graduate (`IMPI_P_GRADED_MIN`, default 0).
- **`fflo/pairbuild.py`**
  - `=` `--coherent-angle-mode` (default `kjac`, prima fisso a kjac).
  - `=` `--qp-gamma-floor` (default NaN, quindi 1e-3 come prima).
  - `+` `--control-analytic`, `--ref-kk-fine-step`, `--residual-taper-qfreeze`, `--residual-qmax-lambda`, passati a pair_gamma solo se diversi da 0.
- **`fflo/pair_gamma.py`**
  - `+` `--ref-kk-fine-step` (KK esatta del riferimento sui suoi bordi).
  - `+` `--residual-taper-qfreeze`.
  - `+` modo taper `qcut_cosine`.
  - `+` `--residual-qmax-lambda` (guardia UV).
  - `=` taper qkin rifattorizzato: q del taper = Q se il congelamento è spento.
- **`fflo/density.py`**
  - `+` `--thouless-q-mode qff-or-zero-fit` (`--zero-fit-qmin/qmax`).
  - `+` `--density-fine-pole` (`occupied_nk_fine`, solo diagnostica).
  - `+` `--density-pole-subtract`, `--sigma0-mode-*`, `--sigma-omega-dense-*`, `--sigma-k-kf-offsets`: alcune c'erano già in FINAL.
- **`test/run_test.py`**
  - manopole nuove per tutte le opzioni sopra;
  - ⚠ default `density-fine-pole = True`: FINAL non l'aveva. È solo diagnostica: cambia n(k) e GAP e non entra nel ciclo. Per rifare FINAL esattamente serve `--no-density-fine-pole`.
  - ⚠ run_test imposta sempre `COHERENT_PHI_PANELS=24` e `COHERENT_PHI_FEATURE_N=21`. Li legge solo il ramo phipanel; con kjac (FINAL) non entrano.

Bug trovati e corretti nel frattempo:
- qcut, fine del taper sbagliata e nome del modo riscritto nella KK: toccava solo t_qcut, che va scartata;
- `test/plot_phase_final.py:p_effective` divide per 2 la correzione della coda η. È solo contabilità dei grafici e non è ancora corretto.

## 3. Regressioni: il codice di oggi con gli argomenti di FINAL rifà FINAL?

Stato su P0p50_final, iterazione 8, ingressi esatti `iter007/next_cubes`:

| prova | risultato |
|---|---|
| Σ (righe dentro il mare, a kF, nella coda, entrambi gli spin) contro `snap/sigma008` | ✓ entro 1-4e-8 |
| bolla e tabella di coppia (15 righe Q) contro la tabella `base` del 2026-10-02, che coincideva con la produzione a 4e-8 | ✓ **identica bit per bit** (ImPiNum, ReInvGamma, ImInvGamma) |
| stadio delle densità (Σ su 28 righe, n(k), densità, contatto, σ0, cube emessa) contro `snap/sigma008` e `iter008/next_cubes` di produzione | ✓ gap ↑/↓ +0.175422% / +0.513188% **identici**, contatto e σ0 a 1e-8 (somme con 2 worker invece di 28); cube identica tranne un nodo dell'asse ω locale in 3-4 righe a k ≥ 21, spostato da quel rumore, con lo stesso integrale di A |

**Conclusione (2026-10-05): il codice di oggi con gli argomenti di FINAL riproduce FINAL.** Ogni differenza delle run successive viene solo dalle ricette della sezione 1.

## 4. Aggiunte del 2026-10-05/06 e ricetta hpfix

Opzioni nuove, tutte spente per default (con gli argomenti di prima il codice fa le stesse cose):

| opzione | dove | cosa fa |
|---|---|---|
| `--qp-e-interp-k2` (run_test) → `QP_E_INTERP_K2=1` | `analytic_bubble.py` (`_interp_e`, energie E1/E2 e geometria della parte coerente), `adn_phi.py` e `impi_table.py` (energia del controllo A0 con `--control-analytic`) | l'energia QP del modello si interpola fra le righe come E − k² e poi si rimette k²; prima si interpolava E linearmente, con errore ~Δk²/4 a k grande |
| `--lattice-n-linear N` (run_test → pairbuild) | `pairbuild.py` | numero fisso di nodi del blocco lineare dei reticoli Ω/ε. Con 0 restano quelli del profilo (turbo: 40 a dw 1e-3, riscalati con dw per tenere ±0.04; a dw 2.5e-4 sono 160) |
| `--coherent-kmax K` (solo pairbuild) | `pairbuild.py` | taglio in k della parte coerente analitica. Con 0 resta Λ. Usata solo dai test della bolla |

Strumenti di controllo (non entrano nel ciclo):
- `test/static_gamma.py`: ReΓ⁻¹(Q, 0) statico esatto dalle cube (asse di Matsubara, Γ0 a η → 0). Serve da arbitro a stato fissato.
- `test/integration_audit/bubble_audit.py` e `compare_pconv.py`, `test/run_bubble.slurm`: varianti della bolla a un passo su uno stato fissato.

**hpfix** (`P0pXX_hpfix`, 2026-10-06) = FINAL2 + `PIN=qff-or-zero-fit` +
```
--taper-qfreeze 1 --ref-kk-fine-step 2.5e-4 --taper-mode qcut_cosine --taper-stop 10 --uv-guard-lambda 2
--alpha 0.2 --control-analytic --qp-e-interp-k2 --coherent-nk 192 --lattice-n-linear 160
```
Il passo del reticolo segue 1 − P, così la banda del minoritario E*_F↓ ~ 0.26 (1 − P) resta coperta da ~100 nodi:

| P | `--lattice-dw` | blocco fine |
|---|---|---|
| 0.90 | 2.5e-4 (quello di FINAL2) | ±0.04 |
| 0.95 | 1.25e-4 | ±0.02 |
| 0.97 | 7.5e-5 | ±0.012 |

Partenze: 0.95 da `P0p95_hp` it19 (copia); 0.97 da `P0p95_hp` it19 con la griglia k rifatta.
Il primo lancio (job 272204-05) è fallito all'avvio perché sul cluster c'era ancora il vecchio run_test.

