#!/bin/bash
# Coda + per ogni run: contact, shift, gap e n(k)k^4 del minoritario per iterazione.
export LC_ALL=C   # altrimenti awk/printf leggono "0.52" come 0 in locale italiano
cd "$(dirname "$0")/.."
echo "======================= CODA ======================="
squeue -u "$USER" -o "%.10i %.16j %.2t %.10M %.10L %R" 2>/dev/null || echo "(squeue non disponibile)"
for L in out/*/loop.log; do
  [ -f "$L" ] || continue
  tag=$(basename "$(dirname "$L")")
  echo; echo "--- $tag ---"
  grep -m1 "k_update_max" "$L" | sed 's/=== //; s/ ===//; s/^/    /'
  grep -h "frozen at" "$L" | head -1 | sed 's/=== //; s/ ===//; s/^/    /'
  awk '
    / PAIR: / { split($0,a,"contact="); split(a[2],b," "); it=$3; c[it]=b[1]
                split($0,s,"shift="); split(s[2],t," "); sh[it]=t[1] }
    / GAP: /  { it=$3; split($0,u,"up="); split(u[2],v," "); gu[it]=v[1]
                split($0,d,"down="); split(d[2],e," "); gd[it]=e[1] }
    / NK4dn: /{ it=$3; x=$0; sub(/.*NK4dn: /,"",x); sub(/ ===.*/,"",x); nk[it]=x }
    / TIME: / { it=$3; tm[it]=$5 }
    END { printf "    %-4s %-10s %-11s %-11s %-11s %-6s %s\n","it","contact","shift","gap_up","gap_down","min","n(k)k^4 dn"
          for (i=1;i<=1000;i++) if (i in c)
            printf "    %-4d %-10s %-11s %-11s %-11s %-6s %s\n", i, c[i], sh[i], gu[i], gd[i], tm[i], nk[i] }' "$L"
  grep -h -E "FAILED|stopping before|LOOP DONE" "$L" | tail -2 | sed 's/^/    /'
done
echo; echo "================== ERRORI =================="
grep -l -E "Traceback|FAILED" out/*/loop.log out/*/iter*/*.log test/logs/*.err 2>/dev/null | head -5 || true
