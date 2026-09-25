#!/bin/bash
# Matrice di test P=0.65 del 2026-09-24.  Uso:  ./submit.sh A|B|C|D|E|F|G|all
# Ogni run parte dal seed warm di seeds/ e si auto-concatena fino a TARGET.
# Tutto cio' che non e' scritto qui e' il default di CONFIG in test/run_test.py
# (e finisce comunque in out/<TAG>/config.json).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
SRC12="--up out/D_good_p12/iter012/next_cubes/A_komega_spinup_iter012.npz --down out/D_good_p12/iter012/next_cubes/A_komega_spindown_iter012.npz"
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
  # come D, ma oltre k_update_max=8 niente seed stale: Sigma analitica di contatto
  # Delta_inf^2/(w + xi_int) (righe risolte a due poli -> n(k) = C/k^4 esatto,
  # anche per le linee interne di Sigma a 8 < p < 12) e densita' = nucleo fino a 8
  # + C/(2*8^2), con C della coppia della stessa iterazione.  In piu' sigma_nomega
  # 97 (il default del motore; qui era 41): i wiggles di n(k)k^4 in [3,8] sono il
  # satellite di Tan campionato su 2-3 nodi omega (test/sigma_resolution_probe.py:
  # errore per riga fino a 15% con 41, <=1.8% con 97, 0.3% con 240).
  G) sub G_good_p12_tail 30 "--pintmax 12 --k-update-max 8 --high-k-sigma pair-contact --contact-tail --sigma-nomega 97" ;;
  # --- un passo solo da D it12 (eccesso gia' uguale nei due spin, +0.014 ciascuno):
  # quale ingrediente di Gamma lo produce?  Uno per job; si legge la densita' FRESCA
  # (alpha=1) con test/onestep_report.py.  Richiede out/D_good_p12/iter012/next_cubes.
  L0) sub L0_base      1 "--pintmax 12 $SRC12" ;;
  L1) sub L1_eta_exact 1 "--pintmax 12 --eta-floor exact_zero $SRC12" ;;
  L2) sub L2_noguard   1 "--pintmax 12 --im-sign-guard off $SRC12" ;;
  # il taper si puo' togliere SOLO con il troncamento corretto: senza, il residuo
  # grezzo vale 1/8 fino a |w|=240 (vuoto mancante fuori dal disco) e il KK misura
  # l'artefatto (test/gamma_ladder_check.py, 2026-09-24).  L4 e' il controllo.
  L3) sub L3_fix_notaper 1 "--pintmax 12 --truncation-fix physical --taper-mode none $SRC12" ;;
  L4) sub L4_fix_taper   1 "--pintmax 12 --truncation-fix physical $SRC12" ;;
  L5) sub L5_delta1e4  1 "--pintmax 12 --delta 1e-4 $SRC12" ;;
  L) for j in L0 L1 L2 L3 L4 L5; do "$0" "$j"; done ;;
  # il loop con il floor eta exact_zero, ripartendo da D it12: L1 a un passo ha
  # portato l'eccesso fresco da +8.1/+1.8% a +1.5/+0.7% a contact invariato.
  # Domanda: il punto fisso regge ed e' stabile?  Confronto diretto con D.
  M) sub M_p12_etaexact 20 "--pintmax 12 --eta-floor exact_zero $SRC12" ;;
  # --- scansione in polarizzazione (2026-09-25).  Configurazione di M (PINTMAX 12,
  # floor eta exact_zero, pair buona), partenza da fermioni LIBERI a mu_avg = 1:
  # mu_up = 1 + P, mu_dn = 1 - P (test/make_free_seed.py).  P = 0.65 dal seed libero e'
  # il controllo contro M (partito da D it12).  --prune-keep 2 tiene le cube solo delle
  # ultime 2 iterazioni (le altre finiscono riassunte in out/<TAG>/snap/).
  # I seed si generano qui: serve l'ambiente python caricato (module + venv).
  P) for P in 0.10 0.20 0.30 0.40 0.50 0.60 0.65 0.70 0.80 0.90; do
       tag="P${P/./p}"
       python3 make_free_seed.py --P "$P" --out-dir "../seeds_free/$tag"
       sub "${tag}_etaexact" 30 "--pintmax 12 --eta-floor exact_zero --seed-dir seeds_free/$tag --prune-keep 2"
     done ;;
  all) for j in A B C D F; do "$0" "$j"; done ;;
  *) echo "uso: ./submit.sh A|B|C|D|E|F|G|L0..L5|L|M|P|all"; exit 1 ;;
esac
