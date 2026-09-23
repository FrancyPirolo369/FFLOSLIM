# FFLO — mappa completa di un ciclo, dal comando ai cubi nuovi

Riferimenti `file:riga` del pacchetto slim (`X_FFLO_SLIM/`).
Convenzioni: `xi_k,s = k^2/m - mu_s` · `G = 1/(w - xi - (ReS - s0) - i ImS + i eta)`
`A = -(1/pi) Im G` · `n_s = int k n(k) dk` (peso lineare, no 2pi) · target `mu_s/2`
`mu_up=1.65  mu_dn=0.35  m=1  =>  kF_up=1.284523  kF_dn=0.591608  qff=0.692915`

```
===============================================================================
 LIVELLO 0 · TU
===============================================================================

  ./submit_long.sh D                    python3 run_fflo.py --out out/x
  (cluster, albero vecchio)             (locale, slim)
       |                                        |
       |  submit_long.sh:4                      |  run_fflo.py:38
       |    sub () { sbatch                     |    CONFIG = [
       |      --export=ALL,TAG="$1",            |      ("out",     "out/run", ...),
       |      TARGET="$2",PMAX="$3",            |      ("iters",   20,        ...),
       |      ALPHA="$4",GLN="${5:-16}",        |      ("alpha",   0.3,       ...),
       |      PAIR_SHIFT_FIXED="${6:-}" \       |      ("pintmax", 6.0,       ...),
       |      --job-name="long_$1"              |      ("q-cap",   20.0,      ...),
       |      submit_long.slurm; }              |      ... 26 manopole ...
       |                                        |    ]
       |  ^ 5 strati sovrapposti,               |  ^ UNA lista. --help le elenca
       |    6 manopole su 12 bloccate           |    tutte con default e motivo.
       |    inline: esportarle non              |    <out>/config.json scritto
       |    aveva effetto, in silenzio          |    PRIMA di calcolare.
       v                                        |
===============================================================================
 LIVELLO 1 · SCHEDULING (solo cluster)
===============================================================================
                                                |
  submit_long.slurm:1                           |
    #SBATCH --cpus-per-task=28                  |
    #SBATCH --mem=120G  --time=04:00:00         |
    #SBATCH --constraint=matrix                 |
    module load python/3.14.3                   |
    source ~/venvs/fflo-sc/bin/activate         |
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1  |
           OPENBLAS_NUM_THREADS=1               |
    ^ il parallelismo e' a PROCESSI:            |  run_fflo.py:242 lo fa da solo
      lasciare i thread BLAS accesi             |    for v in ("OMP_NUM_THREADS",
      = 8 worker x 12 thread su 12 core         |              "MKL_NUM_THREADS",...):
       |                                        |        env.setdefault(v, "1")
       v                                        |
  run_long.sh                                   |
       +----------------------+-----------------+
                              v
===============================================================================
 LIVELLO 2 · RIPRESA E BUDGET
===============================================================================

  scan out/$TAG/iter*/next_cubes
       |
       |  run_long.sh:35-42      ->   run_fflo.py:140
       |    for d in "$D"/iter*/next_cubes; do        def last_complete_iteration(out):
       |      u=$(ls "$d"/*spinup*.npz  | head -1)      for d in sorted(glob.glob(
       |      n=$(ls "$d"/*spindown*.npz| head -1)          os.path.join(out,"iter*",
       |      [ -n "$u" ] && [ -n "$n" ] || continue        "next_cubes"))):
       |      i=$(basename ... | sed 's/iter0*//')        u = glob(".../*spinup*.npz")
       |      [ "$i" -gt "$LAST" ] && LAST=$i            n = glob(".../*spindown*.npz")
       |    done                                        if not u or not n:
       |                                                    continue   # <- monca:
       |  ^ serve ENTRAMBI i cubi. Un job morto a           #    scartata
       |    meta' iterazione lascia un cubo solo:
       |    quella iterazione viene scartata e si
       |    riparte dalla precedente. E' cio' che
       |    rende sicuro il riavvio.
       v
  LAST ---> UP, DN            (seed warm se LAST = 0)
       |
       v
  np.load(UP)  --->  mu_up=1.65   mu_dn=0.35   k[-1]=25.132648
       |
       |  run_fflo.py:201-206
       |    qff = float(np.sqrt(mu_up) - np.sqrt(mu_dn))
       |    cube_k_max = float(np.asarray(zu["k"]).reshape(-1)[-1])
       |    # the short blanket: pairbuild hardcodes --p-int-max 4
       |    q_table_max  = min(cfg["q_cap"], cube_k_max - 4.0 - 0.1)
       |    k_update_max = q_table_max - cfg["pintmax"]
       |
       +--> qff          = 0.6929152795565513
       +--> q_table_max  = 20.0
       +--> k_update_max = 14.0          <=== LA COPERTA CORTA
       |
       |    il 4.0 NON e' configurabile: e' --p-int-max 4 scritto
       |    a mano dentro pairbuild. Il 0.1 e' margine.
       |    Oltre k=14 Sigma non viene MAI aggiornata: resta quella del seed.
       v
  N = min(TARGET - LAST, CHUNK=18)

===============================================================================
 LIVELLO 3 · IL CICLO                                     per i in 1..N
===============================================================================

  griglia q
       |
       |  run_fflo.py:259-263
       |    q_core = smooth_multicenter_grid(0.0, 2.2, cfg["core_n"],
       |                                     [0.0, qff], [0.15, 0.06], [12.0, 15.0])
       |    q_tail = smooth_multicenter_grid(2.2, q_table_max, tail_n,
       |                                     [3.0], [0.6], [6.0])
       |    q_grid = np.unique(np.concatenate([q_core, q_tail]))
       |
       |  run_fflo.py:125   nodi dove serve, invertendo la cumulativa
       |    rho = np.full_like(x, float(base))
       |    for c, w, pk in zip(centers, widths, peaks):
       |        rho += float(pk) / np.cosh((x - float(c)) / float(w)) ** 2
       |    cum = np.concatenate([[0.0],
       |              np.cumsum(0.5*(rho[1:]+rho[:-1]) * np.diff(x))])
       |    return np.interp(np.linspace(0,1,n_points), cum/cum[-1], x)
       |
       |    centri in 0 e in qff: i due posti dove Gamma ha struttura. ~76 nodi.
       v
+-----------------------------------------------------------------------------+
| STADIO 1 · PAIRBUILD          run_fflo.py:266 -> fflo/pairbuild.py           |
+-----------------------------------------------------------------------------+
|                                                                             |
| 1.1  MODELLO QUASIPARTICELLA                                                |
|      pairbuild.py:189-192                                                   |
|        model_up   = qp_model_from_cube(up)                                  |
|        model_down = qp_model_from_cube(down)                                |
|        build_qp_cube(up, qp_up, model=model_up)                             |
|                                                                             |
|      qp_cube.py:81 -> qp_dyson.py:31   cosa c'e' DAVVERO:                   |
|        xi = k[ik] ** 2 / mass - mu                                          |
|        f  = w - xi - (re - sigma0)          # f(E_k) = 0 at the Dyson pole  |
|        sc = np.where(np.diff(np.sign(f)) != 0)[0]     # cambi di segno      |
|                                                                             |
|        t  = -f[j] / df                                                      |
|        wr = w[j] + t * (w[j + 1] - w[j])              # <-- E_k             |
|        # dReSigma/domega from a LOCAL LINEAR FIT over a window around E_k   |
|        dre   = float(np.polyfit(w[sel] - wr, re[sel], 1)[0])                |
|        slope = 1.0 - dre                              # df/dw at the root   |
|        if slope <= 0.0:                                                     |
|            continue            # not a QP (negative residue branch)         |
|        z = min(1.0 / slope, 1.0)                      # <-- Z_k, clip somma |
|        imr = float(np.interp(wr, w, im))                                    |
|        g = z * (abs(imr) + eta)                       # <-- Gamma_k         |
|                                                                             |
|      => il "modello QP" sono TRE ARRAY per spin: E[k], Z[k], Gamma[k]       |
|         + valid[k].                                                         |
|                                                                             |
|      NOTA (docstring qp_dyson.py): i tetti max_z / max_gamma sono stati     |
|      RIMOSSI come criteri di rigetto. Producevano valanghe: un cambiamento  |
|      continuo di Sigma spingeva molti attraversamenti oltre una soglia      |
|      binaria TUTTI INSIEME (Z valido 27% -> 9% in 13 iter; a P=0.65 iter3,  |
|      95 punti persi in un colpo). Spostare il valore spostava solo il       |
|      collasso. Restano solo criteri fisici: slope<=0 e il clip Z<=1.        |
|                       |                                                     |
|                       v                                                     |
| 1.2  BOLLA SU GRIGLIA        pairbuild.py:301                               |
|        command = [sys.executable, "-m", "fflo.impi_table",                  |
|                   "--up-cube-path", str(up), "--down-cube-path", str(down), |
|                   "--q-min","0","--q-max", f"{q_table_max:.15g}", ...]      |
|        ---> impi/impi_residual_grid.npz                                     |
|                       |                                                     |
|                       v                                                     |
| 1.3  BOLLA COERENTE ANALITICA      pairbuild.py:360-372                     |
|        ku, eu, zu, gu, mu_up = _canonical_qp_fields(up, model_up)           |
|        options = dict(angle_mode="kjac", nk=coherent_nk, nphi=64,           |
|                       n_kquad=coherent_nquad,                               |
|                       kmax=float(result["p_int_max"]), gamma_floor=1.0e-3)  |
|        with ProcessPoolExecutor(max_workers=min(workers, q.size),           |
|                                 initializer=_cv_init, ...)                  |
|                                                                             |
|      SCHEMA "MATCHED": la parte coerente e' ANALITICA (dai E,Z,Gamma di     |
|      1.1) e sulla griglia resta solo il residuo. Per questo                 |
|      qp_model_from_cube restituisce UN SOLO modello canonico:               |
|        "Tiny differences between those models do not cancel: they are       |
|         amplified by the pair KK transform."                                |
|        ---> impi/impi_table.npz   =   ImPi(q, Omega)  COMPLETA              |
|                       |                                                     |
|                       v                                                     |
| 1.4  DA PI A GAMMA           pairbuild.py:404                               |
|        command = [sys.executable, "-m", "fflo.pair_gamma",                  |
|                   "--no-plots", "--impi-table-path", str(final_table),      |
|                   "--kk-tail","zero", "--eta-gamma","0.002", ...]           |
|        Kramers-Kronig:  ImPi ---> RePi      (causalita')                    |
|        ---> pair/pair_gamma_table.npz                                       |
|             chiavi:  q , omega , ReInvGamma , ImInvGamma                    |
+------------------------------+----------------------------------------------+
                               |
   run_fflo.py:282-289   log: ReInvGamma(w~0) a Q=0 e a Q=qff, nQ
                               v
+-----------------------------------------------------------------------------+
| STADIO 2 · DENSITY            run_fflo.py:293 -> fflo/density.py main()@1218 |
+-----------------------------------------------------------------------------+
|                                                                             |
| 2.1  SEED          density.py:1360                                          |
|        seeds = {spin: load_seed(path) for spin, path in paths.items()}      |
|        ---> k(406) , w(721) , A , ReS , ImS , sigma0                        |
|                       |                                                     |
|                       v                                                     |
| 2.2  SHIFT DI THOULESS   <=== IL PUNTO CONTESTABILE                         |
|      density.py:1409 -> pair_shifted.py:92, righe 150-155                   |
|                                                                             |
|        if base_shift_override is not None and np.isfinite(...):             |
|            base_shift = float(base_shift_override)          # CONGELATO     |
|        else:                                                                |
|            base_shift = float(np.interp(float(Q_fflo), q,                   |
|                                         re_inv[:, iw0]))    # RITARATO      |
|        shift = base_shift + delta                                           |
|        re_shifted = re_inv - shift    # una costante, su TUTTA la tabella   |
|                                                                             |
|      il regolatore, dove il denominatore si annulla:                        |
|        if floor_mode == "broad":                                            |
|            zero_denom = (np.abs(re_shifted) < eta_floor) &                  |
|                         (np.abs(im_shifted) < eta_floor)                    |
|        else:   # exact_zero: solo il punto di Thouless campionato           |
|            zero_denom = np.hypot(re_shifted, im_shifted) <= 1.0e-14         |
|                       |                                                     |
|                       v                                                     |
| 2.3  CONTACT  <=== IL diagnostico di stabilita' (NON il gap)                |
|      density.py:200                                                         |
|        occupied = omega <= 0.0                                              |
|        delta2 = -float(np.trapezoid(                                        |
|            q * np.trapezoid(a_pair[:, occupied], omega[occupied], axis=1),  |
|            q) / (2.0 * np.pi))                                              |
|        return delta2, 0.25 * delta2          # C = Delta_inf^2 / 4          |
|                                                                             |
|      INTEGRALE DOPPIO su tutta la superficie (Q,Omega). Il riquadro 2.2     |
|      vincola invece UN PUNTO, (qff, 0). Li' sta tutta la discussione.       |
|      Misura: shift congelato -> contact converge a 0.2745.                  |
|              shift ricalcolato -> 0.576 -> 1.081 in 13 iterazioni.          |
|                       |                                                     |
|                       v                                                     |
| 2.4  FINESTRA SICURA      density.py:1478                                   |
|        k_update_max = min(requested_k_max, float(k[-1]))                    |
|        k_mask = k <= k_update_max + 1.0e-12                                 |
|      fuori: Sigma CONGELATA al seed  (--high-k-sigma-mode stale)            |
|                       |                                                     |
|                       v                                                     |
| 2.5  Im SIGMA  <=== IL CUORE                                                |
|      density.py:1520  memory_safe_im_sigma  (ProcessPool, N worker)         |
|        -> sigma_engine.py:342  _im_sigma_row_pair_native                    |
|                                                                             |
|      LA FORMULA (dal docstring):                                            |
|        Im Sigma(k,w) = -pi/(2pi)^2 *                                        |
|          int Q dQ dtheta dOmega A_pair(Q,Omega) A_f(|Q-k|,Omega-w) K(w,Omega)|
|        T=0 kernel  K = +1 su Omega in (0,w),  -1 su (w,0)                   |
|                                                                             |
|      IL TAGLIO ANGOLARE (qui stava un errore fino al 9%):                   |
|        # p(theta) cresce monotonamente da |q-k| a q+k, quindi p<=p_int_max  |
|        # equivale a theta<=theta* con                                       |
|        #   cos(theta*) = (q^2+k^2-p_int_max^2)/(2qk)                        |
|        theta = (np.arange(n_theta)+0.5)[None,:]*theta_star[:,None]/n_theta  |
|        p_th  = np.sqrt(np.maximum(q[:,None]**2 + kv*kv                      |
|                        - 2.0*q[:,None]*kv*np.cos(theta), 0.0))              |
|        theta_weight = 2.0 * theta_star[:, None]              # ESATTO       |
|                                                                             |
|      contro il ramo legacy, che MASCHERAVA su griglia uniforme:             |
|        theta = (np.arange(n_theta)+0.5) * 2.0*np.pi / n_theta               |
|        p_ok  = p_th <= float(p_int_max)      # <- gradino su griglia fissa  |
|        theta_weight = 2.0 * np.pi                                           |
|                                                                             |
|      COSTO: sigma_nk x sigma_nomega = 24 x 41 = 984 valutazioni per spin.   |
|      Tutto il resto e' interpolato. Qui va il tempo.                        |
|                       |                                                     |
|                       v                                                     |
| 2.6  KRAMERS-KRONIG:  Im Sigma ---> Re Sigma                                |
|      density.py:1646 -> kramers_kronig.py:656  kk_re_pv_linear              |
|        m = (y[1:] - y[:-1]) / (x[1:] - x[:-1])                              |
|        c = y[:-1] - m * x[:-1]                                              |
|        seg = mm*(x1-x0) + (mm*w + cc) * np.log(np.abs((x1-w)/(x0-w)))       |
|        seg[ridx, ridx]     = 0.0        # il segmento col polo: PV = 0      |
|        seg[ridx, ridx + 1] = 0.0                                            |
|      ^ PV analitico sull'interpolante lineare a tratti, griglia NATIVA      |
|                                                                             |
|      piu' kramers_kronig.py:877  kk_sigma_gamma_asymptote                   |
|        sottrae la coda n_up * Gamma  (ImSigma ha coda lenta 1/ln^2)         |
|                       |                                                     |
|                       v                                                     |
| 2.7  SIGMA0 = ReSigma(kF, 0)   e la si sottrae                              |
|      sigma_engine.py:799                                                    |
|        k_fermi  = float(np.sqrt(max(float(mass)*float(mu_sigma), 0.0)))     |
|        re_at_kF = np.array([float(np.interp(k_fermi, k, re_mix[:, j]))      |
|                             for j in range(re_mix.shape[1])])               |
|        sigma0   = float(np.interp(0.0, w, re_at_kF))                        |
|                                                                             |
|      "This makes the Luttinger kF coincide with the free kF, so the         |
|       density (and P_meas) come out right by themselves. A +-10%            |
|       k-average biased sigma0 by ~0.013 -> P=0.30 letto 0.316;              |
|       esatto -> 0.303."                                                     |
|                                                                             |
|      => sottrarre sigma0 e' IDENTICAMENTE spostare mu. Pinza la             |
|         superficie di Fermi a kF=sqrt(m mu), e solo allora Luttinger        |
|         impone n = mu/(4pi) e "n vs mu/2" diventa un test.                  |
|                       |                                                     |
|                       v                                                     |
| 2.8  MIXING:   Sigma = alpha*fresh + (1-alpha)*seed        alpha = 0.3      |
|                       |                                                     |
|                       v                                                     |
| 2.9  DYSON      density.py:1846  pole_aware_rebuild                         |
|        sigma_engine.py:971   find_qp_poles_dyson     (trova i poli)         |
|        sigma_engine.py:1124  rebuild_omega_per_k     (infittisce omega)     |
|        sigma_engine.py:718   rebuild_a_from_mixed_sigma_1d:                 |
|                                                                             |
|          xi        = k[:, None] ** 2 / float(mass) - float(mu_sigma)        |
|          sigma_mix = re_shifted + 1j * im_mix                               |
|          g         = 1.0 / (w[None,:] - xi - sigma_mix + 1j*float(eta_g))   |
|          a_mix     = spectral_from_g(g, str(convention))  # A=-(1/pi) Im G  |
|                       |                                                     |
|                       v                                                     |
| 2.10 DENSITA'      density.py:107                                           |
|        def occupied_nk(a, w):        # n(k) = int_{-inf}^{0} A(k,w) dw      |
|            mask = w <= 0.0                                                  |
|            return np.trapezoid(a[:, mask], w[mask], axis=1)                 |
|                                                                             |
|        def row_sum_rule(a, w):       # DEVE fare 1  <- sanity check         |
|            return np.trapezoid(a, w, axis=1)                                |
|                                                                             |
|        def density(k, nk):           # n = int k n(k) dk                    |
|            return float(np.trapezoid(np.asarray(k)*np.asarray(nk),          |
|                                      np.asarray(k)))                        |
|                       |                                                     |
|                       v                                                     |
| 2.11 OUTPUT      density.py:2107   density_summary.txt                      |
|        + blocco  "env SIGMA_PN_* <valore> (env|default)"                    |
|        ^ esiste perche' il 2026-09-21 una ri-esecuzione identica di         |
|          gammafix/iter019 (stessa tabella, stessi seed, diff vuoto su ogni  |
|          parametro registrato) diede una ImSigma diversa del 18% a k        |
|          grande, e la causa non fu identificabile: nessuna di quelle        |
|          variabili era scritta da nessuna parte.                            |
|                       |                                                     |
|                       v                                                     |
| 2.12 CUBI NUOVI   density.py:1979                                           |
|        np.savez(next_cubes/A_komega_spin{up,down}_iterNNN.npz, **cube)      |
|        chiavi: A, ReS, ImS, w, w_base, sigma0, mu_up, mu_dn, mass, eta,     |
|                eps0, pair_contact, pair_delta_inf_squared, mix_alpha,       |
|                iteration, high_k_sigma_mode, high_k_contact_start           |
+------------------------------+----------------------------------------------+
                               |
   run_fflo.py:319-331   log:  PAIR: contact=... (x... vs prec)  shift=...
                               GAP:  up=...  down=...
                               TIME: ... min
                               |
                               |  run_fflo.py:335
                               |    up, down = nu[0], nd[0]
                               |
                               +----------------->  ITERAZIONE i+1
===============================================================================
```

## I tre punti di lettura

**Dove va il tempo** — riquadro 2.5. `sigma_nk x sigma_nomega = 984` valutazioni per
spin. Tutto il resto e' interpolato.

**Dove si decide la convergenza** — riquadro 2.3. Il `pair_contact`, non il gap.
In una run sana contrae dell'1-2% per iterazione; se cresce, la retroazione
`Sigma -> Pi -> Gamma -> Sigma` sta divergendo.

**Dove sta il punto contestabile** — riquadro 2.2. Il ramo `else` ri-tara
l'accoppiamento a ogni giro, vincolando UN punto `(qff, 0)` mentre l'osservabile
del riquadro 2.3 e' un integrale su tutta la superficie `(Q, Omega)`.

## Il filo rosso

`ReSigma - sigma0` compare **due volte**: al riquadro **1.1**, dove i suoi zeri
definiscono `E[k]`, `Z[k]`, `Gamma[k]`; e al riquadro **2.9**, dentro la Dyson.
**E' lo stesso zero.** Il modello QP serve a costruire la bolla coerente (1.3) e a
infittire omega attorno ai poli (2.9): se i due usassero `sigma0` diversi, il ciclo
non si chiuderebbe su se' stesso.

E il budget (`k_update_max = 14`) si decide al **livello 2**, in tre righe di
aritmetica, prima che esista un solo numero di fisica. Con i default sta nel regime
divergente: le misure precedenti danno `<= 8.5` convergente, `>= 11` divergente.
