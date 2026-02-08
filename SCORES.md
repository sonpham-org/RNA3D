# Score Tracking

| # | Date | Score | Approach | Key Change | Notes |
|---|------|-------|----------|------------|-------|
| 0 | pre-Feb7 | 0.359 | TBM-only (Phase 1) | Baseline TBM | Global alignment, geometry constraints, 5 diverse predictions |
| 1 | pre-Feb7 | 0.360 | RNAPro+TBM hybrid | Add RNAPro | N_SAMPLE=1, seed=42, private-best ckpt, TBM templates |
| 2 | Feb 7 | 0.361 | RNAPro+TBM hybrid | N_SAMPLE=5 | 5 diffusion samples instead of 1 (+0.001) |
| 3 | Feb 7 | 0.359 | RNAPro de novo | No templates | Same as TBM-only. Without RibonanzaNet2, RNAPro adds nothing. |
| 4 | Feb 7 | 0.359 | RNAPro+TBM hybrid | Seed=101 | Same as TBM-only. Seed doesn't matter when model is broken. |
