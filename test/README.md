# Test cluster P=0.65 — 2026-09-24

Il core (`fflo/`, `run_fflo.py`) non è modificato. Qui c'è solo:

- `run_test.py`: copia di `run_fflo.py` con le manopole dei test (vedi docstring)
- `submit.sh A|B|C|D|E|F|G|all`: la matrice, da lanciare **da dentro `test/`**
- `run.slurm`: un loop con ripresa, `--time-budget 215` e auto-concatenazione
- `status.sh`: contact, shift, gap e n(k)k⁴ del minoritario per iterazione

Output in `out/<TAG>/` nella radice del repo (già in `.gitignore`).

## Perché questi test

Il punto del 2026-09-24:

1. **La pair turbo gonfia il contact del 33%.** Sullo stesso seed, con `omega=zero`
   (141 nodi) C=0.581; con `zero_and_thresholds` C=0.438. Tutte le run con
   PINTMAX≥6 erano turbo, quindi l'A/B "PINTMAX 3→6" non era pulito.
2. **Λ=4 della bolla taglia `|k_up|`, non l'impulso relativo.** A Q≥10 manca tutto
   il peso di vuoto di ImΠ.
3. **PINTMAX di Σ taglia la coda a k≈PINTMAX+1.5**, dentro la finestra
   `k_update_max = Q_table_max − PINTMAX`.

| job | cosa cambia | domanda |
|---|---|---|
| A | `omega=zero`, PINTMAX 6 | controllo: si riproduce il pompaggio 0.576→1.08? |
| B | pair buona, PINTMAX 6 | con la pair buona il contact pompa ancora? |
| C | B + shift congelato alla prima iterazione | è il re-pinning dello shift? |
| D | pair buona, PINTMAX 12 (k_upd 8) | coda calcolata davvero fino a k=8 |
| E | D + Λ=8, Q_table 17, k_upd 8 | effetto di Λ sul loop (D vs E) |
| F | D + Λ=6 con risoluzione scalata + fix del troncamento | effetto di Λ (pulito) |
| G | D + Σ di contatto analitica e coda C/k⁴ oltre k=8, sigma_nomega 97 | coda pulita e plateau senza wiggles |

**E richiede un `fflo/pairbuild.py` con l'opzione `--bubble-p-int-max`.** È
una modifica del 2026-09-23 non ancora committata. A–D non la usano.

Il criterio di convergenza è il contact (riga PAIR), non il gap.

## Produzione (2026-09-28)

`./submit.sh PROD` (con modulo e venv caricati): per ogni P fino a 0.65 riparte dalla
soluzione convergente della scansione `P0pXX_etaexact` con
la ricetta validata: pair `zero_and_thresholds`, reticolo con 100 nodi in coda, Λ = 4
con taper, `exact_zero`, PINTMAX 12 / k_update 8, Σ 24 k × 97 ω con ring, coda di
contatto analitica oltre k = 8, α = 0.3, `--prune-keep 2`.  Sottoinsieme di P con
`POLS="0.40 0.50" ./submit.sh PROD`.  Correzioni a posteriori da applicare a g_c:
taper/Λ (costruzione alla Enss, ~+0.03–0.04 a P = 0.65) e δ → 0 (≈ −0.013).

## Alta P: Thouless al massimo globale (2026-09-28)

`test/diagnose_highP.py` sulla scansione P: il salto di n_dn a kF_dn (~ Z del
minoritario) crolla con P (0.39 a P = 0.3, 0.12 a 0.65, 0.08 a 0.8) e con lui la
cuspide FFLO di ReΓ⁻¹(Q,0) a qff; il fondo liscio favorisce Q = 0.  Margine
max(Q < 0.8 qff) − picco(qff): −7δ a P = 0.3, −0.5δ a 0.65, +0.3δ a 0.7, +5.9δ a 0.8,
+8.7δ a 0.9.  Col pinning a qff il canale Q ≈ 0 resta supercritico → poli a Ω < 0
(molecole occupate) → C e n_dn scappano (P = 0.8: C = 2.9, n_dn +160%).  Non è
l'integrazione: il peso vero dei poli a Ω < 0 supera quello catturato dalla griglia,
integrare meglio peggiorerebbe.  Anche a P ≤ 0.65 il massimo non è a qff ma a
1.03–1.1 qff, sopra il nodo qff di +4.2δ a P = 0.1, +1.5δ a 0.3, +0.6δ a 0.5, +0.2δ a 0.65
(ripinnando le stesse tabelle: Δg_c = −0.053, −0.018, −0.008, −0.002).

`./submit.sh HI`: ricetta di PROD + `--thouless-q-mode global-max`, due scalette in
parallelo da `out/P0p65_etaexact` (o `HI_SRC`): 0.70 → 0.80 → 0.90 e 0.75 → 0.85
(`HI_CHAIN_A`, `HI_CHAIN_B`; vuota = spenta).  Ogni gradino riparte dall'ultima
iterazione del precedente con `reseed_regrid.py` (griglia k rifatta sui kF nuovi, A
ricostruita con `fflo.density.pole_aware_rebuild`; a P invariato riproduce la densità
della cube entro 0.02 punti percentuali); il passaggio lo fa `run.slurm` (variabili
`LADDER`, `LADDER_ARGS`, `LADDER_SUFFIX`) e si ferma se |gap down| > `LADDER_MAXGAP` = 10%.
Il Q scelto a ogni iterazione è nella riga PAIR di loop.log (colonna Q di status.sh) e
in `snap/iterNNN.npz` (`q_selected`; `shift_used` è ora lo shift applicato da density).

`./submit.sh PRODG`: PROD con `--thouless-q-mode global-max` (anche a P ≤ 0.65 il massimo
non è a qff, vedi sopra).  Ogni P riparte dall'ultima iterazione completa di
`out/P0pXX_prod` se c'è, altrimenti da `out/P0pXX_etaexact`; tag `P0pXX_prodg`.

