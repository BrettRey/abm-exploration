# Project Log

## 2026-02-14 Session 1: First model built and audited

**Duration:** ~1 session, night science mode.
**Model:** Claude Opus 4.6

### What happened

1. **Surveyed the portfolio** for ABM-susceptible claims. Read through ~15 papers. Highest potential: grammaticality (already has code), countability (already has ABM), constructions as mesoscales, deitality.

2. **Started from the grammaticality model** (`simulate_dynamics.py`, 59 lines). That model is single-agent Bayesian evidence accumulation: Beta(a,b) prior, preemption from absence-in-relevant-contexts, two trajectories (rare vs gap). The key move was making it multi-agent so preemption becomes endogenous.

3. **Built v1** (200 agents, random mixing). Six constructions spanning core to novel. Results:
   - Core/established lock in, stable gap dies, community novel dies slowly. Replicates single-agent predictions.
   - **Surprise: rare_licensed goes bimodal.** Community splits into accepters and rejecters. Not predicted.
   - Critical mass sweep found tipping point at ~15% initial users (for rho=0.3, niche=0.10).
   - Niche sweep confirmed: wider niche = faster death (fan-out plot).

4. **Added network structure** (v2, 1000 agents, 10 communities). Clustering experiment:
   - **Pre-registered prediction:** clustering saves rare forms by lowering critical mass.
   - **Result: opposite.** Random mixing spreads the form best. Clustering creates dialectal pockets (user community thrives at 77% confidence, others die at 24%). Prediction was wrong, which was informative.

5. **Gelman checkpoint.** Brett asked what Gelman would say. Answer: document the garden of forking paths, write the generative model as math not code, prior predictive simulation, pre-commit predictions, sensitivity analysis. Wrote `MODEL_SPEC.md`.

6. **rho sensitivity audit** (6 values × 20 MC runs per cell, 151s total). This was the big one:
   - rho gates everything. Critical mass ≈ rho/(1+rho).
   - Clustering effect **reverses sign** across rho: random wins at medium rho, clustering preserves pockets at high rho.
   - Bimodality is a phase-transition marker, not a general property.
   - Updated MODEL_SPEC.md with full results table.

### Decisions made

- Raw Python, not Mesa. Scripts, not notebooks.
- Gelman-style discipline even in night science: MODEL_SPEC.md, pre-registered predictions, sensitivity audits.
- rho is the key parameter. Don't trust any result without checking rho sensitivity.

### What's unresolved

- rho is free and unestimated. The model is an intuition pump until rho is pinned.
- Sensitivity sweeps for tau, beta, N, C, initial_confidence not yet run.
- No generational turnover, no analogical support, no written evidence, no competition between constructions.
- Haven't touched deitality yet (the original target domain).
- The 54% cell (rho=0.3, f=0.08): stable mixed equilibrium or bistable? Would need longer runs.

7. **Mean-field analysis** (`models/mean_field.py`). Derived the critical mass analytically:
   - Lower bound: f > rho/(1+rho)
   - Mean-field estimate: f > rho / (c_prod * (1+rho))
   - Both are **conservative**: the simulation is more forgiving at every rho, especially low rho (2% seed survives where the bound predicts 5-9%).
   - The mean-field trajectory doesn't capture the **positive-feedback cascade**: in the ABM, converted non-users become producers and amplify the signal. The deterministic mean-field has non-users monotonically declining.
   - Key practical result: the formula inverts to estimate rho from data: rho = f* * c_prod / (1 - f* * c_prod). If we knew the real critical mass for a construction, we could pin rho.

### Decisions made

- Raw Python, not Mesa. Scripts, not notebooks.
- Gelman-style discipline even in night science: MODEL_SPEC.md, pre-registered predictions, sensitivity audits.
- rho is the key parameter. Don't trust any result without checking rho sensitivity.

### What's unresolved

- rho is free and unestimated. The model is an intuition pump until rho is pinned. (But we now have a formula to pin it if we find empirical critical mass data.)
- The mean-field misses the cascade dynamics. A stochastic mean-field might do better.
- Sensitivity sweeps for tau, beta, N, C, initial_confidence not yet run.
- No generational turnover, no analogical support, no written evidence, no competition between constructions.
- Haven't touched deitality yet (the original target domain).
- The 54% cell (rho=0.3, f=0.08): stable mixed equilibrium or bistable? Would need longer runs.

## 2026-02-14 Session 2: Data exploration for empirical rho estimation

**Model:** Claude Opus 4.6

### What happened

8. **Surveyed available databases** for empirical (frequency, acceptability) pairs. Searched for resources that combine gradient acceptability judgments with corpus frequency for the same constructions. Found ~15 resources; most have one or the other, not both.

9. **Downloaded MegaAcceptability v2** (White & Rawlins, megaattitude.io). 375,000 ratings for 1,007 verbs × 50 syntactic frames. Gradient 1-7 Likert scale. Freely available.

   Analysis found:
   - **Variance peaks in the contested zone** (mean 3-4: SD=2.16) vs the extremes (mean 1-2: SD=0.70; mean 6-7: SD=1.08). Consistent with ABM prediction that community disagreement peaks near the tipping point.
   - **But Sarle's bimodality coefficient does NOT peak in the middle.** BC peaks for clearly accepted items (6-7: BC=0.688), because those have most raters at 6-7 with a few outliers at 1-2. The contested zone has high variance but spread across the whole scale, not bimodal. The ABM's bimodality prediction is about community-level production, not individual rating distributions, so this may not be the right test.
   - **Conceptual problem:** MegaAcceptability tests verb-frame selectional restrictions. A verb that can't take a frame was probably NEVER grammatical in that frame. Our model is about constructions that COULD survive but die from insufficient community support. Wrong phenomenon.

10. **Tried to download Scots Syntax Atlas data** (scotssyntaxatlas.ac.uk). 258 syntactic features, 147 locations, 500+ speakers, Likert-scale judgments + 275 hours spoken corpus. API exists (`/api/v1/`) but SiteGround anti-bot protection triggered after ~1 successful request. Download script written with resume capability (`data/scots/download_all.py`), needs manual run with longer delays or email to scotssyntaxatlas@gmail.com for bulk access.

    **Why the Scots data is the right dataset:** dialect features genuinely vary across communities (some spreading, some stable, some dying), both acceptance AND production are measured in the same population, and different features are at different stages of vitality. This matches what the model needs far better than MegaAcceptability.

11. **Downloaded White & Rawlins (2020) analysis notebook** (frequency-acceptability-selection). They used VALEX corpus for verb-frame frequencies. Key finding: frequency explains <1/3 of acceptability variance. "Acceptability is necessary but not sufficient for usage." Some verb-frames are acceptable but never used ("latent constructions").

### What we learned

- The MegaAcceptability data is the wrong grain for estimating rho: it's about argument structure selection, not construction survival in a community.
- The Scots Syntax Atlas is the right grain: genuinely variable dialect features with both judgment and production data.
- The bimodality prediction from the ABM is about community-level production distributions, not individual rating distributions. Testing it requires geographic/community variation, not just inter-rater variance.

### What's unresolved

- Scots data not yet downloaded (rate-limited). Need to retry with delays or request bulk access.
- rho still unestimated. The Scots data could provide it if we can pair feature acceptance rates with feature production rates across communities.
- Haven't read Francis (2022) Ch. 5 on the frequency-acceptability relationship.

### Possible next moves

- Get the Scots data (email or patient download)
- Read Francis (2022), *Gradient Acceptability and Linguistic Theory*, Ch. 5
- Try YGDP data (Yale Grammatical Diversity Project) cross-referenced with COCA
- Sweep remaining ABM parameters (tau, within_bias)
- Port the model to deitality
- Park it and think
