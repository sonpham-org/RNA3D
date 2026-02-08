# CLAUDE.md - Project Rules

## Kaggle Submission Rules
- **ALWAYS set `"enable_internet": false`** in kernel-metadata.json for competition submissions
- Competition notebooks run with internet disabled during scoring - submissions with internet enabled cannot be immediately submitted
- All model weights, packages, and data must be pre-uploaded as Kaggle datasets

## Project Structure
- `kaggle_submissions/` - All submission notebooks, each in its own directory
- Each submission directory contains: notebook `.ipynb` + `kernel-metadata.json`
- `PROBLEM.md` - Competition problem description and data structure
- `STRATEGY.md` - Explore/exploit submission strategy
- `SCORES.md` - Score tracking table (update after every submission)

## Key Files
- RNAPro source: `kaggle_submissions/phase2_rnapro/RNAPro/`
- Base notebook: `kaggle_submissions/phase2_rnapro/phase2-rnapro-tbm-hybrid.ipynb`

## Competition
- Stanford RNA 3D Folding Part 2
- Metric: TM-score (higher = better)
- Deadline: March 25, 2026
- 5 submissions per day, notebook-only, no internet during scoring
