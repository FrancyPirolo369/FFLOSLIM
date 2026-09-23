# X_FFLO_SLIM — la pipeline senza il morto

Copia ridotta di `X_FFLO_FULLGRID_PROD` contenente **solo il codice che il loop vivo
può effettivamente raggiungere**, più un unico driver con tutti i default in un posto.

**L'albero originale non è toccato.** I runner legacy continuano a funzionare, e
questa cartella si rigenera da zero con un comando. Niente è stato tagliato a mano.

## Quanto è stato tolto

```
modulo                      orig   tolte   resta     %
--------------------------------------------------------
sigma_engine                4923    3735    1188   75%
impi_table                  2609       0    2609    0%
density                     2136       0    2136    0%
feature_grid                2214    2039     175   92%
proxy                       1821    1323     498   72%
kramers_kronig              1705     682    1023   40%
sigma_from_pair             1589     637     952   40%
adn_phi                     1274     346     928   27%
sigma_test                  1165    1008     157   86%
pair_gamma                  1133       0    1133    0%
spectral                    1019     707     312   69%
impi_cv                      784     647     137   82%
decompose_qp_inc             585     298     287   50%
analytic_bubble              546       0     546    0%
pair_shifted                 506     311     195   61%
pairbuild                    473       0     473    0%
compare_motors               379     268     111   70%
lorentzian_cube              274     210      64   76%
lattice_grids                228     113     115   49%
qp_cube                      194      48     146   24%
qp_dyson                     130       0     130    0%
--------------------------------------------------------
TOTALE                     25687   12372   13315   48%
```

Il pezzo più grosso è il motore Σ: **3735 righe su 4923 (75%)**, di cui 614 di solo
`parse_args`. Quel codice è il driver autonomo del motore, usato dai runner vecchi
(`run_sc_from_control_poleaware.sh`, `critical_line/run_critical_line.sh`, una
dozzina di `.slurm`) — non dal loop attuale, che lo usa **come libreria** e ne importa
12 simboli su 53.

Gli "0%" sono script con `main()`: tutto è raggiungibile *da lì*. Hanno comunque rami
e modalità mai esercitati, ma toglierli richiede un'analisi diversa (env mai
impostate, `choices` di argparse mai usate) — è una seconda fase.

## Come si usa

```bash
cd X_FFLO_SLIM

# vedi la configurazione risolta senza far girare niente
python3 run_fflo.py --dump-config --out out/prova

# due iterazioni dal seed warm
python3 run_fflo.py --out out/prova --iters 2 --workers 10

# riprendi: stesso comando, riparte dall'ultima iterazione completa
python3 run_fflo.py --out out/prova --iters 5 --workers 10

# la run D: accoppiamento congelato
python3 run_fflo.py --out out/gfix --iters 20 --pair-shift-fixed -0.019825

# giro veloce (4x): il costo e' sigma_nk * sigma_nomega
python3 run_fflo.py --out out/smoke --iters 1 --sigma-nk 12 --sigma-nomega 21 \
                    --n-theta 16 --core-n 20 --tail-n 10
```

`python3 run_fflo.py --help` elenca **ogni** manopola con default e motivazione.

### Cosa cambia rispetto a prima

| prima | adesso |
|---|---|
| 5 strati (default Python, `loop_wide.py`, `run_long.sh`, `submit_long.sh`, la tua shell via `--export=ALL`) | una lista `CONFIG` in `run_fflo.py` |
| 6 manopole su 12 assegnate **inline** in `run_long.sh`: esportarle non aveva effetto, in silenzio | ogni manopola è un flag |
| la configurazione non era scritta da nessuna parte | `<out>/config.json` prima di qualunque calcolo, incluse le derivate (`qff`, `q_table_max`, `k_update_max`, valutazioni di Σ per spin) |
| `grep -c` produceva `"0\n0"` e rompeva il confronto | non c'è più |
| `PAIR_SHIFT_FIXED` sopravviveva alla ri-sottomissione solo per via di `--export=ALL` | è un flag, passa o non passa |

## I seed

`seeds/` contiene i due cubi warm con cui parte il loop, copiati da
`X_FFLO_FULLGRID_PROD/cluster_bundle_20260922/seeds/` (2026-09-22):

```
b11b35f9163c62a0754a05f505863ee955a170af12a55c3533f0e42ff79db01a  A_komega_spindown_warm.npz
97b6658ae504a0f00b45316338dcc337d37757432d255268cd650f8a22f62d34  A_komega_spinup_warm.npz
```

Sono qui come file veri, non come link, così `git clone` basta per far girare
la pipeline. Per partire da altri cubi: `--up`/`--down`, oppure `--seed-dir`.

## Rigenerare

```bash
python3 build_slim.py --clean
```

Fa un punto fisso su *(modulo → nomi richiesti da quel modulo)*: parte dai quattro
entry point, calcola con `ast` quali simboli top-level sono raggiungibili, guarda
quali `from X import a, b` sopravvivono dentro il codice vivo, e itera. Poi riscrive
gli import sul namespace piatto `fflo.`, applica le patch dichiarate in `PATCHES`, e
**verifica importando ogni modulo** e controllando che ogni simbolo richiesto ci sia.

L'analisi è volutamente conservativa — un nome nudo, un attributo, una callback
passata per riferimento o una menzione a livello di modulo contano tutti come uso —
quindi **sotto**stima il codice morto. Ciò che dichiara morto lo è.

Se un sorgente a monte cambia in modo che una `PATCHES` non si applichi più, il build
**si ferma con un errore** invece di produrre silenziosamente una copia sbagliata.

## NON ANCORA FATTO: il collaudo numerico

L'analisi statica garantisce che il codice rimosso sia irraggiungibile, e la verifica
garantisce che tutto si importi. **Non** garantisce che i numeri siano identici: resta
scoperto il dispatch dinamico (un `getattr`, un nome preso da una tabella di
configurazione), che nessuna analisi AST vede.

**Prima di usare questo per fisica** va girato uno stadio density con la stessa
tabella di coppia e gli stessi seed, qui e nell'albero originale, e confrontati i
`density_summary.txt`. Finché quel confronto non è verde, questa cartella è una
ripulitura strutturale verificata, non un motore validato.

## Un avvertimento sui default

Con i default (`q_cap=20`, `pintmax=6`) viene fuori `k_update_max = 14`. Dalle misure
precedenti: `k_update_max ≤ 8.5` converge, `≥ 11` diverge. **Il default sta nel regime
divergente.** È così anche nel bundle originale — non è una regressione introdotta
qui — ma adesso il numero è stampato in `config.json` invece di essere implicito.
