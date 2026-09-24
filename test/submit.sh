#!/bin/bash
# Matrice di test P=0.65 del 2026-09-24.  Uso:  ./submit.sh A|B|C|D|E|all
# Ogni run parte dal seed warm di seeds/ e si auto-concatena fino a TARGET.
# Tutto cio' che non e' scritto qui e' il default di CONFIG in test/run_test.py
# (e finisce comunque in out/<TAG>/config.json).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
sub () {  # TAG TARGET ARGS
  sbatch --export=ALL,TAG="$1",TARGET="$2",ARGS="$3" --job-name="slim_$1" run.slurm
}
case "${1:-}" in
  # controllo: la pair turbo di prima (omega=zero).  Deve riprodurre il
  # pompaggio del loop cluster del 22/09 (contact 0.576 -> ~1.08 in 13 iter).
  A) sub A_turbo_p6      12 "--omega-mode zero --pintmax 6" ;;
  # la domanda principale: con la pair buona il contact pompa ancora?
  B) sub B_good_p6       30 "--pintmax 6" ;;
  # stesso, accoppiamento congelato al valore della SUA prima iterazione:
  # separa il ri-pinning dello shift dal resto
  C) sub C_good_p6_frz   30 "--pintmax 6 --pair-shift-freeze" ;;
  # PINTMAX coerente con la finestra: 12 >= k_update_max(8) + 3, quindi la
  # coda e' calcolata davvero fino a k=8 e non tagliata a k~7.5
  D) sub D_good_p12      30 "--pintmax 12" ;;
  # come D ma bolla con Lambda=8: il deficit di vuoto parte a Q~16 invece
  # che a Q~8.  D vs E = effetto di Lambda sul loop
  # ATTENZIONE: E mescola Lambda e risoluzione (31 nodi p fissi su [0,8]) e non ha
  # la correzione del troncamento.  Superato da F.
  E) sub E_good_p12_L8   30 "--pintmax 12 --bubble-lambda 8 --k-update-max 8" ;;
  # come D ma con la bolla fatta bene: Lambda=6 con risoluzione scalata (Delta p
  # e densita' coerente come a Lambda=4) e il peso libero fuori dal disco
  # aggiunto a mu fisico (test/fix_truncation.py).  Scan del 2026-09-24:
  # Lambda 6 vs 8 cosi' differiscono <= 2.8 delta su ReGamma^-1(Q,0).
  F) sub F_good_p12_L6fix 30 "--pintmax 12 --bubble-lambda 6 --p-nodes 46 --coherent-nk 144 --truncation-fix physical --k-update-max 8" ;;
  all) for j in A B C D F; do "$0" "$j"; done ;;
  *) echo "uso: ./submit.sh A|B|C|D|E|F|all"; exit 1 ;;
esac
