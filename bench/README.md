# Sheet-switch benchmark

Per-node on-sheet / off-sheet labels for 14 public PHerc segments. Use them to score your own sheet-switch detector, or to
audit the truth behind Turncoat's Evidence table.

```bash
turncoat bench score --flags MY_FLAGS/          # or: python bench/score_detector.py --flags MY_FLAGS/
turncoat bench score --flags bench/flags/turncoat         # node exact 0.537, fibre pairs 0.234, verified nodes 0.416
turncoat bench score --flags bench/flags/tifxyz-doctor    # 0.032 / 0.041 / 0.020
```

## Files

| file | content |
|---|---|
| `manifest.json` | per mesh: scroll, split, public segment id and tifxyz URL, the hi-res registration the truth was made on, grid shape, sha256 of `x.tif`, counts |
| `labels/KEY.png` | uint8 grid, same shape as the segment's tifxyz: **1** on sheet, **0** off sheet, **255** unlabelled |
| `fibre_pairs.csv` | `mesh,row_a,col_a,row_b,col_b,relation`: node pairs one automatic fibre relates, `same` or `different` sheet |
| `verified_nodes.csv` | `mesh,row,col,label`: nodes where the truth and the fibres agree |
| `human_fibre_nodes.csv` | `mesh,row,col,label`: labels from human-guided fibres alone, no tracing (PHerc0139, 230 nodes) |
| `flags/turncoat/`, `flags/tifxyz-doctor/` | the flags behind the README's numbers, in the input format below |

| key | scroll | split | segment (9.4 µm tifxyz) | labelled nodes | off sheet |
|---|---|---|---|---|---|
| w025 | PHerc0139 | test | [20250108000000-w025_2025010863](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0139/segments/20250108000000-w025_2025010863/mesh/20250108000000-on-20250728140407-9.362um.tifxyz/) | 73,909 | 3.2 % |
| w028 | PHerc0139 | test | [20250108000003-w028_2025010836](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0139/segments/20250108000003-w028_2025010836/mesh/20250108000003-on-20250728140407-9.362um.tifxyz/) | 78,285 | 6.2 % |
| w029 | PHerc0139 | test | [20250108000004-w029_2025010827](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0139/segments/20250108000004-w029_2025010827/mesh/20250108000004-on-20250728140407-9.362um.tifxyz/) | 77,655 | 6.1 % |
| w034 | PHerc0139 | test | [20260303000000-w034_2026030317](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0139/segments/20260303000000-w034_2026030317/mesh/20260303000000-on-20250728140407-9.362um.tifxyz/) | 63,579 | 27.0 % |
| p5a | PHerc0500P2 | test | [20250611154256-z_dbg_gen_00354_inp_hr](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0500P2/segments/20250611154256-z_dbg_gen_00354_inp_hr/mesh/20250611154256-on-20250820143440-9.362um.tifxyz/) | 9,794 | 62.7 % |
| p5b | PHerc0500P2 | test | [20250716055231-z_dbg_gen_00473_inp_hr](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0500P2/segments/20250716055231-z_dbg_gen_00473_inp_hr/mesh/20250716055231-on-20250820143440-9.362um.tifxyz/) | 10,144 | 55.7 % |
| p5c | PHerc0500P2 | test | [20250611132918-z_dbg_gen_00302_inp_hr](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0500P2/segments/20250611132918-z_dbg_gen_00302_inp_hr/mesh/20250611132918-on-20250820143440-9.362um.tifxyz/) | 9,573 | 52.8 % |
| p814b | PHerc0814 | test | [20250925212701-auto_grown_20250925212701145](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0814/segments/20250925212701-auto_grown_20250925212701145/mesh/20250925212701-on-20250804134230-9.362um.tifxyz/) | 19,362 | 65.2 % |
| p814c | PHerc0814 | test | [20250929040153-auto_grown_20250929040153260](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0814/segments/20250929040153-auto_grown_20250929040153260/mesh/20250929040153-on-20250804134230-9.362um.tifxyz/) | 28,819 | 47.9 % |
| p814e | PHerc0814 | test | [20251001060526-auto_grown_20251001060526760](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0814/segments/20251001060526-auto_grown_20251001060526760/mesh/20251001060526-on-20250804134230-9.362um.tifxyz/) | 48,439 | 29.9 % |
| p841a | PHerc0841 | test | [20260220213127-w00](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0841/segments/20260220213127-w00/mesh/20260220213127-on-20250821151531-9.366um.tifxyz/) | 15,450 | 64.1 % |
| p841b | PHerc0841 | test | [20260220214732-auto_grown_20260220144552896](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0841/segments/20260220214732-auto_grown_20260220144552896/mesh/20260220214732-on-20250821151531-9.366um.tifxyz/) | 14,802 | 72.1 % |
| w045 | PHerc0139 | dev | [20260126000000-w045_2026012619](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0139/segments/20260126000000-w045_2026012619/mesh/20260126000000-on-20250728140407-9.362um.tifxyz/) | 77,766 | 34.6 % |
| w046 | PHerc0139 | dev | [20260325000000-w046_20260325](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0139/segments/20260325000000-w046_20260325/mesh/20260325000000-on-20250728140407-9.362um.tifxyz/) | 77,244 | 36.0 % |

Rows and columns are those of the published 9.4 µm tifxyz (`x.tif` byte-identical, sha256 in the manifest). Turncoat's
rule was fitted on the two dev meshes; compare detectors on the 12 test meshes.

## How the truth was made

- **Same segment, sharper scan.** Each segment also has a registration on its scroll's 2.2-2.4 µm scan (URL in the
  manifest), on the same grid. There the sheets are resolved.
- **Automatic tracing.** We traced each node's sheet automatically on that registration. These labels are machine made,
  not checked by a person.
- **Off sheet** (0) = the node is not on the majority sheet of its spiral turn. **Unlabelled** (255) = the tracing gave
  the node no sheet (41 % of test nodes, 29 % of dev nodes). Unlabelled nodes are never scored.

## Checks of the truth

- **Automatic fibres (independent of the truth and of Turncoat).** Qual's AFV fibre model V2 and the AVF spline
  generator (pinned Hugging Face revisions) ran in 84 random 256³ regions of the hi-res scans, 6 per mesh, drawn with seed
  20261026 without looking at the truth or any detector. On node pairs a fibre relates, the truth agrees with the fibres
  on 82.0 % (PHerc0139), 75.6 % (PHerc0500P2), 76.3 % (PHerc0814) and 75.8 % (PHerc0841). `verified_nodes.csv` keeps only
  nodes where every such pair agrees.
- **Human-guided fibres (PHerc0139 only, small).** Used alone, they label 230 nodes on w028, w029 and w034
  (`human_fibre_nodes.csv`). Where the truth also labels the node, it agrees on all 43 nodes the fibres put on sheet, and
  on 7 of the 97 they put off sheet. Anchored on the truth instead, the same fibres agree with 560 of 571 truth-off nodes
  on the PHerc0139 meshes here, but also call 286 of 374 truth-on nodes off. Their off-sheet vote does not separate the
  two classes, so treat that check as weak.
- **Rules fixed in advance.** The test meshes, the truth method and the node-exact rule were frozen before any truth or
  detector run on the test meshes. Turncoat's rule was frozen in [../PREREG.md](../PREREG.md) before its one scoring run.

## Limits

- The labels are the output of an automatic tracing, not a human ground truth. Disagreement with the automatic fibres is
  18-24 % of related pairs.
- Off-sheet shares of 30-72 % on PHerc0500P2, PHerc0814 and PHerc0841 are high for published segments. Nothing
  independent of the fibres checks how much of that is truth error.
- The labels use a per-turn majority. If most of a turn is on the wrong sheet, that turn's labels flip.
- Four scrolls only: the ones with a public hi-res scan and segments registered on both scans. None of them is on the
  First Letters eligible list (checked 2026-10-27).

## Inspect the labels in VC3D

Download a segment's tifxyz folder (link above), then:

```bash
turncoat bench vc3d PATH/TO/20260303000000-on-20250728140407-9.362um.tifxyz --mesh w034
```

This adds two layers to the segment's `ink-detections.json`: **Benchmark truth: off sheet** and **Benchmark truth:
unlabelled**. In VC3D's **Ink Detection** dock, pick one with Colormap **Fire** (marked nodes show orange). Ctrl+click a
spot to see the slices there. The command refuses a folder whose grid does not match the labels.

## Score your detector

1. Write one file per mesh into a folder, named by key (`w034.png`, ...), on the tifxyz grid (rows x columns):
   - `KEY.png`: 8-bit, nonzero = flagged;
   - `KEY.npy`: bool or int;
   - `KEY.npz`: an array named `flags`;
   - `KEY.tif`: float, > 0.5 = flagged;
   - or Turncoat's output folder `KEY/suspicion.tif`.
2. Run `turncoat bench score --flags FOLDER [--json result.json]`. Use `--meshes w034,p5a` for a subset, and
   `--split dev` or `--split all` for the other meshes.
3. The scorer reports R / P / F1 pooled over the test meshes and per scroll, for each test:
   - **node exact**: tp = flagged label-0 nodes, fp = flagged label-1 nodes;
   - **fibre pairs**: a pair counts as called when exactly one of its nodes is flagged, and the positives are
     `different` pairs;
   - **verified nodes**: node exact on `verified_nodes.csv`;
   - **human-fibre nodes**: node exact on `human_fibre_nodes.csv`.

   Pooling sums the counts over meshes.

To share a result, open an issue or PR with the flag folder (or the JSON output) and the commit of your detector.
