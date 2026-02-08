# Explore/Exploit Submission Strategy

## Philosophy

We treat this as a **multi-armed bandit problem**. Each submission slot is a pull.
- **Explore:** Try fundamentally new approaches, configurations, or models
- **Exploit:** Iterate on what's already scoring well
- **Budget:** 5 submissions/day × ~46 days = ~230 total pulls
- **Feedback loop:** Each submission returns a score within hours

### Allocation Rule (adaptive)
- **Week 1-2:** 80% explore / 20% exploit (cast wide net)
- **Week 3-4:** 50% explore / 50% exploit (refine promising directions)
- **Week 5-6:** 20% explore / 80% exploit (converge on best approach)
- **Final week:** 100% exploit (polish best configuration)

---

## Current Weaknesses (Score: 0.360)

1. **Circular templates:** TBM predictions → RNAPro templates is noisy bootstrapping
2. **Single diffusion sample:** N_SAMPLE=1 per template run (should be 5+)
3. **Wasted 5-prediction slots:** All 5 predictions are slight variations of same template
4. **No de novo capability:** Entirely dependent on template availability
5. **512 token crop:** RNAPro silently crops long sequences, losing structure info
6. **No model quality assessment:** No ranking/selection of best predictions
7. **Binary merge:** RNAPro OR TBM, no confidence-weighted blending
8. **Single checkpoint:** Only using private-best, not ensembling checkpoints

---

## Submission Waves

### WAVE 1: Quick Wins (Days 1-3, ~15 submissions)
Goal: Exhaust easy improvements to current approach before exploring new directions.

#### 1A. Fix RNAPro sampling (HIGH PRIORITY)
- Change `N_SAMPLE=5` (currently 1) → 5 diverse diffusion samples per run
- Use ranking_score to pick best sample for each of the 5 prediction slots
- Expected impact: +0.02-0.05 TM-score (more diversity = better best-of-5)

#### 1B. Multi-seed RNAPro
- Run with `--seeds 42,101,202` instead of just `--seeds 42`
- Each seed × 5 samples = 15 candidate structures, pick best 5
- Expected impact: +0.01-0.03

#### 1C. Better template combinations
- Currently: 5 separate runs with template_idx 0-4 (subsets of same TBM templates)
- Better: Run once with N_SAMPLE=5, multiple seeds, pick 5 most diverse good structures
- Use confidence scores for selection instead of fixed template combos

#### 1D. Template quality sweep
- Try RNAPro WITHOUT templates (de novo mode): `--use_template None`
- Compare: templates vs no-templates to see if our TBM templates help or hurt
- If templates hurt → our template generation is the bottleneck

#### 1E. Both checkpoints
- Try `RNAPro-Public-Best-500M` vs `RNAPro-Private-Best-500M`
- They may excel on different target types
- Ensemble: use public-best for some targets, private-best for others

### WAVE 2: Template Revolution (Days 4-8, ~25 submissions)
Goal: Templates are the #1 factor. Dramatically improve template quality.

#### 2A. john's 1st-place TBM approach
- The gold standard for template generation in Part 1
- Available as public notebook: https://www.kaggle.com/code/jaejohn/rna-3d-folds-tbm-only-approach
- Study and implement his exact method
- Expected impact: +0.05-0.10 (template quality is the biggest lever)

#### 2B. MMseqs2-based template search
- Use structural database search instead of just sequence alignment
- Available notebook: https://www.kaggle.com/code/rhijudas/mmseqs2-3d-rna-template-identification
- May find better templates than sequence-only search

#### 2C. Improved alignment scoring
- Current: simple global alignment with fixed parameters
- Try: local alignment, profile-based alignment, structure-aware scoring
- Use RibonanzaNet2 embeddings for similarity (odat's approach from 2nd place)

#### 2D. Template ensembling
- Generate templates from MULTIPLE methods (TBM + MMseqs2 + embedding-based)
- Feed best template from each method to different prediction slots
- This diversifies the 5 predictions meaningfully

#### 2E. Per-target template analysis
- For each test target, measure template quality (alignment score, coverage)
- Route: good template → TBM/RNAPro-with-template, bad template → de novo
- Adaptive rather than one-size-fits-all

### WAVE 3: De Novo & Alternative Models (Days 9-16, ~40 submissions)
Goal: Part 2 has template-free targets. Need de novo prediction capability.

#### 3A. RNAPro de novo (no templates)
- Run RNAPro with `--use_template None --use_msa true`
- With RibonanzaNet2 + MSA only
- Test on targets where we suspect no good templates exist

#### 3B. Integrate DRFold2
- Used by 1st place winner (john) in Part 1
- Install and run as separate prediction pipeline
- Ensemble with RNAPro predictions

#### 3C. Try RhoFold+
- RNA-specific model with language model backbone
- Pre-trained on 23.7M RNA sequences
- Good de novo predictor, no template dependency
- Complementary to RNAPro's template-heavy approach

#### 3D. Secondary structure → 3D pipeline
- Generate 2D structure predictions (RNAfold, EternaFold, etc.)
- Feed as constraints to structure prediction
- trRosettaRNA2 showed this can beat AF3

#### 3E. Protenix base model
- The AF3 reproduction that RNAPro is built on
- Try it directly on RNA targets
- Different training data/approach may complement RNAPro

### WAVE 4: Ensembling & Selection (Days 17-24, ~40 submissions)
Goal: Combine predictions from multiple methods optimally.

#### 4A. Model Quality Assessment (MQA)
- Implement lociPARSE or ARES for RNA structure scoring
- Score all candidate structures, pick best 5 per target
- This is how the "agentic tree search" approach hit 0.635

#### 4B. Diversity-weighted selection
- From N candidate structures, select 5 that are:
  (a) individually high-quality (by MQA score)
  (b) mutually diverse (cover different structural hypotheses)
- This optimizes the "best of 5" scoring metric

#### 4C. Per-target routing
- Classify each target: has-template vs template-free, short vs long, single vs multi-chain
- Route to best method per category
- Build a routing table from score feedback

#### 4D. Confidence-weighted blending
- Instead of binary RNAPro-or-TBM, blend coordinates weighted by confidence
- Residue-level blending: use RNAPro for confident regions, TBM for others

#### 4E. Cross-method ensembling
- Generate predictions from: RNAPro (template), RNAPro (de novo), TBM, DRFold2, RhoFold+
- Use MQA to rank all candidates
- Select top 5 diverse structures across all methods

### WAVE 5: Refinement & Polish (Days 25-35, ~55 submissions)
Goal: Squeeze out remaining score from best approach.

#### 5A. Physics-based refinement
- Apply BRiQ energy minimization to top predictions
- Fix steric clashes, improve backbone geometry
- Small but consistent improvement

#### 5B. RNA constraint optimization
- Better bond length/angle constraints (current: 5.95Å for C1'-C1')
- Literature values for RNA backbone geometry
- Apply as post-processing to all predictions

#### 5C. Hyperparameter optimization
- Sweep: N_cycle (4, 8, 10, 15), N_step (50, 100, 200), N_sample (5, 10, 20)
- For each, measure quality vs runtime trade-off
- Find sweet spot for Kaggle GPU time limit

#### 5D. Long sequence handling
- For sequences > 512 nt: sliding window with overlap
- Predict overlapping segments, stitch together
- Or: predict at 512, then refine full-length with constraints

#### 5E. Multi-chain assembly
- Predict individual chains separately, then dock
- Use inter-chain constraints from coevolution signals
- May beat concatenated-sequence approach for complexes

### WAVE 6: Final Push (Days 36-46, ~55 submissions)
Goal: Lock in best approach and polish.

#### 6A. Best configuration lock
- By now we know which methods/configs score best
- Run the winning configuration with maximum compute budget
- Fill all 5 prediction slots optimally

#### 6B. Target-specific tuning
- For targets where we're scoring poorly, try specialized approaches
- More diffusion samples, different templates, different models

#### 6C. Submission format optimization
- Ensure coordinate precision is optimal
- Verify no NaN/zero residues are dragging score down
- Sanity check all multi-chain assemblies

---

## Score Tracking Protocol

After EVERY submission, record in `SCORES.md`:
```
| Date | Sub# | Score | Approach | Key Change | Notes |
```

Use this to:
1. Identify which changes actually help vs hurt
2. Decide explore vs exploit for next submission
3. Track diminishing returns per direction
4. Build intuition for what the test set looks like

### Decision Rules
- If a change improves score by >0.01: **EXPLOIT** (iterate on this direction)
- If a change is neutral (±0.005): **Note and move on**
- If a change hurts by >0.01: **Revert and EXPLORE** different direction
- If stuck for 3+ submissions: **Force exploration** of completely new approach

---

## Resource Constraints & Time Budget

### Per Submission (~9 hour GPU limit on Kaggle):
| Component | Estimated Time | Notes |
|-----------|---------------|-------|
| TBM generation | ~5-15 min | Fast, CPU-bound |
| RNAPro setup | ~5 min | Copy files, install deps |
| RNAPro inference (short seqs) | ~2-4 hrs | 512-token limit, 10 cycles, 200 steps |
| RNAPro inference (all seqs) | ~4-8 hrs | Depends on N_sample × seeds |
| Post-processing | ~5 min | Merge, format, validate |
| **Buffer** | ~1 hr | For errors, retries |

### Optimization levers for time:
- Reduce N_step (200→100) for 2x speedup with moderate quality loss
- Reduce N_cycle (10→6) for ~40% speedup
- Process only promising targets with full settings
- Use mini/tiny for rapid prototyping (not for final submission)

---

## Immediate Next Actions

### Today (Day 1):
1. **Sub 1:** Fix N_SAMPLE=5 in current approach (quick win)
2. **Sub 2:** RNAPro de novo (no templates) to establish baseline
3. **Sub 3:** Multi-seed (42, 101, 202) with current templates

### Tomorrow (Day 2):
4. **Sub 4:** Try public-best checkpoint instead of private-best
5. **Sub 5:** Begin implementing john's 1st-place TBM
6. Start studying john's public notebook in detail

### Day 3:
7-9. Template quality experiments based on Day 1-2 results
10. First ensemble attempt (if multiple approaches show promise)
