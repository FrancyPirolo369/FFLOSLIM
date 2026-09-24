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
| G | D + Σ di contatto analitica e coda C/k⁴ oltre k=8 | la coda stale conta? (D vs G) |

**E richiede un `fflo/pairbuild.py` con l'opzione `--bubble-p-int-max`.** È
una modifica del 2026-09-23 non ancora committata. A–D non la usano.

Il criterio di convergenza è il contact (riga PAIR), non il gap.
