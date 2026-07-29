#!/bin/bash
# Harvest the Si EXX+RPA EOS series from rusty into the json the fitter reads.
#
#   ./eos_exxrpa_harvest.sh > eos_exxrpa.json
#   python3 eos_exxrpa_fit.py eos_exxrpa.json
#
# Only points whose run actually printed an "RPA energy:" line are emitted --
# a run that died in the Pi stage leaves a truncated rpa.out with a plausible
# but partial "Total energy", and silently fitting those is how you get a
# confident wrong lattice constant.
set -u
ROOT=${ROOT:-'~/ceph/CoQui/abinit/eos_exxrpa'}
ssh rusty "
first=1
echo '{'
for a in 10.05 10.15 10.25 10.35 10.45 10.55; do
  f=$ROOT/a\$a/rpa.out
  [ -f \"\$f\" ] || continue
  ec=\$(grep -h '^RPA energy:' \"\$f\" | tail -1 | awk '{print \$3}')
  [ -n \"\$ec\" ] || continue
  # 'Total energy:              <value> a.u.'  -> field 3.
  # Do NOT regex-scrape the line: '[-0-9.]+' also matches the dots in 'a.u.'.
  tot=\$(grep -h '^Total energy:' \"\$f\" | tail -1 | awk '{print \$3}')
  [ -n \"\$tot\" ] || continue
  [ \$first -eq 1 ] || echo ','
  first=0
  printf '  \"%s\": {\"total\": %s, \"ec\": %s}' \"\$a\" \"\$tot\" \"\$ec\"
done
echo
echo '}'
"
