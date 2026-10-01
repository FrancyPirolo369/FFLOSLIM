#!/bin/bash
# Test di ricetta (2026-09-30): quale ingrediente di PROD sposta g_c e C rispetto a etaexact
# (Dg -0.14 ... -0.30, DC -20 ... -40%), e PROD e' convergente nei parametri numerici?
#
# Ogni variante riparte dall'ultima iterazione completa di out/P0pXX_prod (convergita), fa 2
# iterazioni con alpha = 1 e cambia UNA manopola.  Al punto fisso fresco = seme, quindi il
# controllo T0 deve restare fermo; lo scarto di una variante da T0 e' la risposta a un passo
# (il punto fisso nuovo sta circa 1/(1-J) ~ 3 volte piu' in la').  Manopole della coppia
# (T1, R3, R4) si vedono gia' all'iterazione 1; quelle di Sigma (T2-T5, R1, R2) all'iterazione 2.
#
#   T0_ctrl      ricetta PROD invariata
#   T1_tail20    reticolo con coda del profilo (turbo 20) invece di 100      [etaexact]
#   T2_som41     Sigma su 41 nodi omega invece di 97                          [etaexact]
#   T3_stale     Sigma oltre k_update_max congelata (stale) invece di contatto [etaexact]
#   T4_notail    senza coda analitica C/k^4 in n(k)                           [etaexact]
#   T5_etaexact  tutte e quattro insieme = ricetta etaexact
#   R1_som193    Sigma su 193 nodi omega (passo attorno a omega = 0: 7e-3 -> 3.5e-3)
#   R2_snk48     Sigma su 48 k invece di 24 (righe vicino a kF)
#   R3_dw        reticolo eps/Omega fine, passo 2.5e-4
#   R4_pnodes    griglia p della bolla raddoppiata (p-nodes 62, coherent-nk 192)
#   R5_pnodes4   griglia p della bolla x4 (p-nodes 124, coherent-nk 384): D -> 0? (R4 porta D 0.88 -> 0.39)
#   R6_lam6      Lambda della bolla 6 con nodi scalati (passo come R4: p-nodes 93, coherent-nk 288) +
#                --truncation-fix physical (raccomandazione del 24/09 mai entrata in prod): R6 vs R4 isola
#                Lambda + fix a parita' di risoluzione vicino a kF.  Nota: q_table_max scende a ~19
#   R7_gold      preset di quadratura gold della bolla: finestre di nodi attorno a kF_up, kF_dn e |Q+-kF|
#                (larghezza 0.05, 7 nodi), coherent-nk 448 / nquad 160, 81 nodi omega di feature.
#                Riferimento per R4/R5, che danno C e g di segno opposto (griglia lineare senza nodi a kF)
#   R8_qff       i due rimedi al massimo oltre qff trovati con le ricostruzioni a posteriori (2026-09-30):
#                griglia p della residua fine (p-nodes 124, coherent-nk 384, come R5; D 0.88 -> 0.31) +
#                Sigma su 193 nodi omega (come R1: dimezza il passo attorno a 0, cioe' il pavimento
#                ImSigma(kF,0) ~2-3e-3 che domina la larghezza QP a kF; D 0.88 -> 0.58 togliendone il 75%)
#
# Uso (da test/):   DRY=1 ./submit_ablation.sh      (stampa e basta)
#                   ./submit_ablation.sh            P=0.50 di default
#                   P=0.30 ONLY="T0_ctrl R1_som193" ./submit_ablation.sh
#                   CONTINUE=1 TARGET=8 ONLY="T0_ctrl T2_som41 T5_etaexact" ./submit_ablation.sh
#                     -> riprende varianti gia' esistenti fino a TARGET iterazioni (run.slurm riparte
#                        dall'ultima iterazione completa; il seme non viene ricopiato)
# Tag: P0pXX_abl_<nome> (li prende anche il solito rsync di P0p*).  Lettura: test/ablation_report.py
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs ../seeds_warm

module load python/3.14.3
source ~/venvs/fflo-sc/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

P="${P:-0.50}"
DRY="${DRY:-0}"
CONTINUE="${CONTINUE:-0}"
TARGET="${TARGET:-2}"
tag="P${P/./p}"
src="../out/${tag}_prod"

# 0) differenze di configurazione registrate fra etaexact e prod (config.json di run_test)
python3 - "$tag" <<'EOF'
import json, os, sys
tag = sys.argv[1]
a, b = (os.path.join("..", "out", f"{tag}_{f}", "config.json") for f in ("etaexact", "prod"))
if os.path.exists(a) and os.path.exists(b):
    ca, cb = json.load(open(a)), json.load(open(b))
    skip = {"out", "up", "down", "target", "iters", "workers", "time_budget", "prune_keep", "seed_dir"}
    diff = [k for k in sorted(set(ca) | set(cb)) if not k.startswith("_") and k not in skip and ca.get(k) != cb.get(k)]
    print(f"== config etaexact -> prod ({tag}): " + (", ".join(f"{k}: {ca.get(k)} -> {cb.get(k)}" for k in diff) or "nessuna differenza"))
else:
    print(f"== config.json mancante per {tag}_etaexact o {tag}_prod: confronto saltato")
EOF

read -r it up down < <(python3 -c "import run_test
last, paths = run_test.last_complete_iteration('$src')
assert paths, 'nessuna iterazione completa in $src'
print(last, paths[0], paths[1])")
seed="../seeds_warm/${tag}_abl"
echo "== seme: $(basename "$src") iterazione $it -> ${seed#../}"
if [ "$DRY" != 1 ] && [ "$CONTINUE" != 1 ]; then
  mkdir -p "$seed"
  cp "$up" "$seed/A_komega_spinup_reseed.npz"
  cp "$down" "$seed/A_komega_spindown_reseed.npz"
fi

COMMON="--pintmax 12 --k-update-max 8 --eta-floor exact_zero --prune-keep 2 --alpha 1"
TAIL="--lattice-n-tail 100"; SOM="--sigma-nomega 97"; HK="--high-k-sigma pair-contact"; CT="--contact-tail"
declare -A V=(
  [T0_ctrl]="$COMMON $TAIL $SOM $HK $CT"
  [T1_tail20]="$COMMON --lattice-n-tail 0 $SOM $HK $CT"
  [T2_som41]="$COMMON $TAIL --sigma-nomega 41 $HK $CT"
  [T3_stale]="$COMMON $TAIL $SOM --high-k-sigma stale $CT"
  [T4_notail]="$COMMON $TAIL $SOM $HK"
  [T5_etaexact]="$COMMON --lattice-n-tail 0 --sigma-nomega 41 --high-k-sigma stale"
  [R1_som193]="$COMMON $TAIL --sigma-nomega 193 $HK $CT"
  [R2_snk48]="$COMMON $TAIL $SOM $HK $CT --sigma-nk 48"
  [R3_dw]="$COMMON $TAIL $SOM $HK $CT --lattice-dw 2.5e-4"
  [R4_pnodes]="$COMMON $TAIL $SOM $HK $CT --p-nodes 62 --coherent-nk 192"
  [R5_pnodes4]="$COMMON $TAIL $SOM $HK $CT --p-nodes 124 --coherent-nk 384"
  [R8_qff]="$COMMON $TAIL --sigma-nomega 193 $HK $CT --p-nodes 124 --coherent-nk 384"
  [R7_gold]="$COMMON $TAIL $SOM $HK $CT --profile gold"
  [R6_lam6]="$COMMON $TAIL $SOM $HK $CT --bubble-lambda 6 --p-nodes 93 --coherent-nk 288 --truncation-fix physical"
)
ORDER=(T0_ctrl T1_tail20 T2_som41 T3_stale T4_notail T5_etaexact R1_som193 R2_snk48 R3_dw R4_pnodes)
# R5_pnodes4, R6_lam6, R7_gold e R8_qff non sono nell'elenco di default: si lanciano con ONLY="..."
for name in ${ONLY:-${ORDER[@]}}; do
  [ -n "${V[$name]:-}" ] || { echo "variante sconosciuta: $name"; exit 1; }
  run="${tag}_abl_${name}"
  if [ -d "../out/$run" ] && [ "$CONTINUE" != 1 ]; then echo "  $run: out/$run esiste gia', salto"; continue; fi
  if [ ! -d "../out/$run" ] && [ "$CONTINUE" = 1 ]; then echo "  $run: non esiste, niente da riprendere"; continue; fi
  args="${V[$name]} --up ${seed#../}/A_komega_spinup_reseed.npz --down ${seed#../}/A_komega_spindown_reseed.npz"
  cmd=(sbatch --export=ALL,TAG="$run",TARGET="$TARGET",ARGS="$args" --job-name="slim_$run" run.slurm)
  if [ "$DRY" = 1 ]; then
    echo "  DRY: ${cmd[*]}"
  else
    env -u LADDER -u LADDER_ARGS -u LADDER_SUFFIX "${cmd[@]}"
    echo "  $run"
  fi
done
