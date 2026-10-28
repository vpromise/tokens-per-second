#!/bin/zsh
# Re-render finished episodes after a kit change (cached voice, no narrator needed for unchanged lines):
# briefs/rerender.sh 7 9 10 ...   prints old and new durations and the layout result
for n in "$@"; do
  d=$(print -l ~/videos/inference-series/ep$(printf %02d $n)-*(/N) | head -1)
  s=$(grep -m1 -oE "^class Ep[0-9]+[A-Za-z0-9]+" $d/scene.py | cut -d' ' -f2)
  before=$(afinfo $d/$s.mp4 | grep -oE "estimated duration: [0-9.]+" | cut -d' ' -f3)
  (cd $d && manimgx render scene.py > rerender.log 2>&1 && manimgx present scene.py --no-open >> rerender.log 2>&1)
  after=$(afinfo $d/$s.mp4 | grep -oE "estimated duration: [0-9.]+" | cut -d' ' -f3)
  echo "ep$n $s: $before s -> $after s; $(grep -m1 -oE 'layout: [^|]*' $d/rerender.log); spoke: $(grep -c 'speak' $d/rerender.log)"
done
