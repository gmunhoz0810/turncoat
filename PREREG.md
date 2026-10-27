# Detector pre-registration

Frozen 2026-10-26 15:45 ET in the development repository (commit 230ce01), before the detector was scored on any of the
12 test meshes. Reproduced verbatim below, except passages that name private modules or private comparison arms, which
are marked [redacted]. The rule in this repository (`src/turncoat/detector.py`) produces bit-identical flags to the
frozen code on all 12 test meshes, the 2 dev meshes and 4 earlier meshes.

---

**Code**: [redacted: development-repo module path] at **3d6dcdd**. Uses only dinovol (community model, MIT) through
Turncoat's profile tracker, mesh geometry (umbilicus-based spiral turn index, normals) and Turncoat v3 leave / step alarms.
[redacted: list of private modules it does not import]

**Fit (dev only, w045 + w046)**: 192 settings (per-turn reference on / off, a2 0.2-0.45, region 1-10 mm2, v3 support
0 / 1 / 5 / 20); rule: the highest pooled node-exact F1 against the per-turn hi-res truth. Selected: whole-mesh reference,
a2 >= 0.3, regions >= 1 mm2, >= 1 v3 alarm hit within 1 mm. Dev F1 0.906 (R 88.8 %, P 92.5 %).

**Score once** on the 12 fresh meshes: node-exact R / P / F1 against the hi-res truth (pooled, per scroll), fibre pairs
only (AFV, no [redacted]), and nodes where truth and fibres agree; against [redacted] and tifxyz-doctor.

---

The full fit table is `evidence/dev_fit.json`; the one-time scores are `evidence/fresh12_node_exact.json` and
`evidence/fibre_check.json`.
