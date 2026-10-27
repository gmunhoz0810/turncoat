# Evidence index

Where every number in the README, the video, the form answer and the Discord post comes from. `python evidence/score.py`
recomputes all rates from the stored counts; `pytest tests/test_evidence.py` checks they match.

| number | meaning | source |
|---|---|---|
| F1 0.537, R 47.8 %, P 61.4 % | Turncoat, node exact, 12 held-out meshes pooled | `evidence/fresh12_node_exact.json` summary.fresh12.turncoat |
| F1 0.032, R 1.6 %, P 71.6 % | tifxyz-doctor, same scoring | same file, summary.fresh12.tifxyz-doctor |
| 449,811 scored nodes | sum of off + on over the 12 meshes | same file, per_mesh |
| 12 meshes, 4 scrolls | PHerc0139 w025 w028 w029 w034; PHerc0500P2 p5a p5b p5c; PHerc0814 p814b p814c p814e; PHerc0841 p841a p841b | same file, per_mesh.*.scroll |
| per scroll F1 0.499 / 0.461 / 0.470 / 0.768 (doctor 0.002 / 0.090 / 0.006 / 0.074) | node exact per scroll | same file, summary.PHerc0139 ... PHerc0841 |
| F1 0.234 (R 14.0 %, P 69.9 %) vs 0.041 | fibre pairs, 22,662 two-sheet pairs | `evidence/fibre_check.json` fresh12.fibre_pairs |
| F1 0.416 (R 38.3 %, P 45.6 %) vs 0.020 | 3,614 nodes where truth and fibres agree, 1,246 off-sheet | same file, fresh12.verified_nodes |
| truth agrees with fibres on 82.0 % (PHerc0139), 75.6 / 76.3 / 75.8 % (0500P2 / 0814 / 0841) | truth_pair_agree / truth_pairs per scroll | same file, per_mesh |
| 84 regions of 256³ | fibre regions, 6 per mesh, seed 20261026 | same file, "what" |
| dev F1 0.906 (R 88.8 %, P 92.5 %), 192 settings | rule fit on w045 + w046 | `evidence/dev_fit.json` (first row is the selected rule) |
| "finds only 1-2 % of off-sheet points" | doctor recall 1.6 % node exact, 2.1 % fibre pairs, 1.0 % verified nodes | the two files above |
| "misses about half" | 1 - recall 47.8 % = 52.2 % | `evidence/fresh12_node_exact.json` |
| w025: 129 of 197 flagged scored nodes false | per_mesh.w025.turncoat tp 68, fp 129 | same file |
| about 15 min on a laptop, under 3.5 GB | full `turncoat check` on published w034 (117,276 nodes) on an Apple-silicon MacBook: 923.7 s, peak RSS 3.30 GB | measured 2026-10-27 (summary.json "seconds" of that run) |
| 3 min 36 s in the video | the same check with the leave / step alarms cached, 32-vCPU Linux box | the screen recording |
| 4-15 min per segment on one RTX 4090 | dinovol embeddings, 213-883 s measured per mesh | development log, 2026-10-25 |
| 27 regions, 34,131 flagged nodes on w034 | output of the demo run | `turncoat check` printout in the video |

The truth: each node's sheet, traced automatically on the same segment's registration on its scroll's 2.2-2.4 µm scan;
a node is off-sheet when it is not on the majority sheet of its spiral turn. The rule was frozen in `PREREG.md` before the
one scoring run. The fibres come from AFV fibre model V2 and the AVF spline generator (pinned Hugging Face revisions),
which Turncoat does not use.
