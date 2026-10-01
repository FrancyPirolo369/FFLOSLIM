#!/bin/bash
# Coda + per ogni run: contact, shift, gap e n(k)k^4 del minoritario per iterazione.
# Q0-qff = ReGamma^-1(Q=0, 0) - ReGamma^-1(qff, 0) in unita' di delta = 1e-3 (gara fra i canali:
# > 0 vince Q = 0, pBCS; < 0 vince il ramo FFLO).
export LC_ALL=C   # altrimenti awk/printf leggono "0.52" come 0 in locale italiano
cd "$(dirname "$0")/.."
echo "======================= CODA ======================="
squeue -u "$USER" -o "%.10i %.16j %.2t %.10M %.10L %R" 2>/dev/null || echo "(squeue non disponibile)"
for L in ${STATUS_GLOB:-out/*/loop.log}; do
  [ -f "$L" ] || continue
  tag=$(basename "$(dirname "$L")")
  echo; echo "--- $tag ---"
  grep -a -m1 "k_update_max" "$L" | sed 's/=== //; s/ ===//; s/^/    /'
  grep -ah "frozen at" "$L" | head -1 | sed 's/=== //; s/ ===//; s/^/    /'
  awk '
    / PAIR: / { split($0,a,"contact="); split(a[2],b," "); it=$3; c[it]=b[1]
                split($0,s,"shift="); split(s[2],t," "); sh[it]=t[1]
                if (index($0, " Q=")) { split($0,g," Q="); split(g[2],h," "); qs[it]=h[1] } }
    / GAP: /  { it=$3; split($0,u,"up="); split(u[2],v," "); gu[it]=v[1]
                split($0,d,"down="); split(d[2],e," "); gd[it]=e[1] }
    / NK4dn: /{ it=$3; x=$0; sub(/.*NK4dn: /,"",x); sub(/ ===.*/,"",x); nk[it]=x }
    / TIME: / { it=$3; tm[it]=$5 }
    /ReInvGamma\(w~0\)/ { it=$3; split($0,z,"Q=0 -> "); split(z[2],z0," ")
                split($0,f,"Q=qff -> "); split(f[2],f0," ")
                mg[it]=sprintf("%+.1f", (z0[1]-f0[1])/1e-3) }
    END { printf "    %-4s %-10s %-11s %-7s %-7s %-11s %-11s %-6s %s\n","it","contact","shift","Q","Q0-qff","gap_up","gap_down","min","n(k)k^4 dn"
          for (i=1;i<=1000;i++) if (i in c)
            printf "    %-4d %-10s %-11s %-7s %-7s %-11s %-11s %-6s %s\n", i, c[i], sh[i], qs[i], mg[i], gu[i], gd[i], tm[i], nk[i] }' "$L"
  grep -ah -E "FAILED|stopping before|LOOP DONE" "$L" | tail -2 | sed 's/^/    /'
done
echo; echo "================== ERRORI =================="
grep -l -E "Traceback|FAILED" out/*/loop.log out/*/iter*/*.log test/logs/*.err 2>/dev/null | head -5 || true
