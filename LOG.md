# RNA3D Competition Log

Tracks every session, what was done, what was learned, and what's next.

---

## Session 1 — Pre-Feb 7: Baseline Setup

**What we did:**
- Set up project structure: `PROBLEM.md`, `STRATEGY.md`, `SCORES.md`
- Built Phase 1 TBM-only baseline (`phase1_tbm/`)
  - Global pairwise alignment (BioPython) to find similar training sequences
  - Template adaptation via alignment mapping + gap interpolation
  - RNA geometry constraints (bond lengths, angles, self-avoidance)
  - 5 diverse predictions via: best template, jitter, hinge, chain perturbation, smooth wiggle
- Built Phase 2 RNAPro+TBM hybrid (`phase2_rnapro/`)
  - TBM generates templates → converted to .pt → fed to RNAPro as `ca_precomputed`
  - RNAPro runs with N_SAMPLE=1, seed=42, N_CYCLE=10, N_STEP=200
  - Merge: RNAPro for short sequences (<=1000nt), TBM fallback for long sequences
  - Private-best checkpoint only

**Scores:**
- Phase 1 TBM-only: **0.359**
- Phase 2 RNAPro+TBM hybrid: **0.360**

**Learnings:**
- RNAPro adds barely +0.001 over raw TBM — our templates are the bottleneck, not the model
- The circular bootstrap (TBM predicts → feeds RNAPro as template) is essentially giving RNAPro noisy versions of the same TBM answer
- Need to break this cycle with better templates or go fully de novo

---

## Session 2 — Feb 7: Wave 1 Quick Wins

**What we did:**
- Created `sub1_nsample5/`: Changed N_SAMPLE from 1 to 5 (5 diffusion samples per run)
  - Same templates, same seed=42, same checkpoint
  - RNAPro now generates 5 diverse structures per template instead of 1
- Created `sub2_denovo/`: RNAPro with `--use_template None` (de novo, no templates)
  - Tests whether RibonanzaNet2 + MSA alone can beat our noisy TBM templates
  - Falls back to TBM for long sequences (>1000nt)
- Created `sub3_multiseed/`: Seed=101 (A/B test against sub1's seed=42)
  - Same N_SAMPLE=5, same templates, just different random seed
  - Tests seed sensitivity — if score differs, multi-seed ensembling is valuable

**Scores:**
- Sub 1 (N_SAMPLE=5): **0.361** (+0.001 from baseline)
- Sub 2 (de novo): **not submitted** (needs investigation)
- Sub 3 (seed=101): **not submitted** (needs investigation)

**Learnings:**
- N_SAMPLE=5 gave a tiny +0.001 improvement — sampling helps but templates still dominate
- Sub2/Sub3 were not submitted to Kaggle cloud — need to debug the push process
- All 3 kernel-metadata.json files are missing RibonanzaNet2 in model_sources — this means RNAPro runs WITHOUT RibonanzaNet2 embeddings (falls back to `use_RibonanzaNet2 false`). This could be a significant missed improvement.

**Open issues:**
1. ~~Sub2 and Sub3 need to be pushed/submitted to Kaggle~~ DONE — slug mismatch was the issue (title slug != id slug). Fixed and pushed both.
2. RibonanzaNet2 model source missing from all kernel-metadata.json — need to add it
3. Current approach ceiling seems low — templates are the limiting factor

---

## Session 3 — Feb 7: Research Team Sprint

**What we did:**
- Assembled 8-agent research team (PI, comp-bio, 3 ML researchers, creative-thinker, resource-finder, verifier)
- Pushed sub2_denovo and sub3_multiseed to Kaggle (fixed slug mismatch issue)
- Full research sprint covering: RNA biology, existing tools, dataset strategies, modeling, creative ideas, resources
- Verifier fact-checked 23 key claims: 15 verified, 7 minor corrections, 0 wrong
- PI compiled comprehensive `RESEARCH_PLAN.md` (389 lines, 4-tier roadmap)

**Key findings:**
1. **RibonanzaNet2 is missing** from all kernel-metadata.json — RNAPro runs as generic AF3, not RNA-specialized. 1-minute fix.
2. **Circular template pipeline** — all 5 templates are variations of the same alignment. Need genuinely diverse templates.
3. **Wasteful inference** — 5 passes x 1 sample instead of 1 pass x 5 samples. Fixing saves ~60% GPU time.
4. **john's TBM** scored 0.591 with same concept as ours (0.359). The 0.232 gap is all template quality.
5. **RNAPro scored 0.648** on Part 1 retrospectively (better than we thought).
6. **36%+ test targets are RNA-protein complexes** — co-folding could be significant.
7. **All planned tools verified real** — DRFold2, RhoFold+, lociPARSE, Protenix all have public code/weights.

**Deliverables:**
- `RESEARCH_PLAN.md` — 4-tier roadmap: 0.361 → 0.40 → 0.48 → 0.53 → 0.55+
- Verified resource catalog (models, datasets, notebooks)
- Per-tier experiment plan with time estimates

## Next Steps (Tier 0 — Immediate)
1. Add RibonanzaNet2 to kernel-metadata.json `model_sources`
2. Rewrite inference: 1 pass with N_SAMPLE=5 (not 5 passes x 1 sample)
3. Multi-seed (42, 101, 202)
4. Try both checkpoints (public-best + private-best)
5. Study john's TBM notebook for Tier 1
