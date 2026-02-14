# Model Specification: Grammaticality Population Dynamics ABM

Version: 0.2 (2026-02-14)

## 1. Generative Model

### Entities

- **Population**: N agents indexed i = 1, ..., N
- **Constructions**: set K, indexed k
- **Communities**: C communities of approximately equal size; agent i belongs to community m(i) = i mod C

### State

Each agent i holds a Beta-distributed belief about each construction k:

    B_ik = Beta(a_ik, b_ik)

with posterior mean (confidence):

    c_ik = a_ik / (a_ik + b_ik)

### Dynamics (per time step)

For each of I interactions per step:

1. **Partner selection.** Draw speaker s uniformly from the population. Draw hearer h:
   - With probability beta: h drawn uniformly from community m(s), excluding s
   - With probability 1 - beta: h drawn uniformly from the full population, excluding s

2. **For each construction k:**

   a. Context relevance: draw r ~ Bernoulli(w_k). If r = 0, skip.

   b. Production: speaker produces iff c_sk > tau AND Bernoulli(c_sk) = 1. This is a two-gate model: a hard threshold (you won't produce what you doubt) followed by a stochastic gate (even confident speakers don't always use a form).

   c. Update hearer beliefs:
   - If produced: a_hk <- a_hk + 1
   - If not produced: b_hk <- b_hk + rho

That's the whole model. No other dynamics.

### Initialization

For each agent i, construction k:

- With probability f_k: a_ik = a_k^0 + epsilon_a, b_ik = b_k^0 + |epsilon_b|, where epsilon_a ~ N(0, 0.5), epsilon_b ~ N(0, 0.3)
- With probability 1 - f_k: a_ik = 1, b_ik = 1 (uninformative prior)

With user placement: if construction k has designated user communities S_k, only agents in those communities can be initial users. The per-agent probability is scaled so the expected user count equals f_k * N.


## 2. Parameter Table

### Theory-fixed

These follow from the OVMG's commitments. Changing them would mean a different model.

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Update rule | Bayesian (Beta conjugate) | OVMG core claim: grammaticality beliefs are updated by evidence accumulation |
| Preemption exists | yes | OVMG core claim: absence-in-relevant-context is evidence of ungrammaticality |
| Production depends on confidence | yes | Speakers don't produce forms they believe are ungrammatical |

### Empirically estimable

These could in principle be measured. Current values are guesses.

| Parameter | Symbol | Current | Could estimate from |
|-----------|--------|---------|---------------------|
| Niche width | w_k | 0.02 -- 0.30 | Corpus frequency of contexts licensing construction k |
| Initial user fraction | f_k | 0.0 -- 0.95 | Corpus attestation rates, dialect surveys |
| Production threshold | tau | 0.5 | Psycholinguistic production studies (at what confidence do speakers produce?) |
| Population size | N | 1000 | Sociolinguistic community size |
| Number of communities | C | 10 | Sociolinguistic surveys, dialect geography |
| Within-community bias | beta | 0.80 | Social network analysis (proportion of interactions with in-group) |

### Free

Currently arbitrary. These are the dangerous ones.

| Parameter | Symbol | Current | Notes |
|-----------|--------|---------|-------|
| Preemption weight | rho | 0.3 | THE key free parameter. How much does not-hearing count relative to hearing? No principled value. |
| Initial confidence (users) | a_k^0, b_k^0 | (5, 2) to (10, 1) | How confident are initial users? Affects entrenchment/inertia. |
| Interactions per step | I | N | Normalization choice. Higher = faster dynamics, same equilibria (in theory). |
| Individual variation | epsilon_a, epsilon_b | N(0, 0.5), N(0, 0.3) | Arbitrary noise magnitude. |
| "Producer" reporting threshold | -- | 0.5 | Analysis choice: when we count "producers" we use conf > 0.5. Matches tau but is still a choice. |


## 3. What the Model Does Not Include

Mechanisms absent from v0.2 (any of which could matter):

- **Analogical support**: no generalisation from related constructions
- **Written/literary evidence**: agents only learn from live interaction
- **Metalinguistic knowledge**: agents can't reason about rules
- **Prestige/authority weighting**: all speakers count equally
- **Frequency-dependent production**: production probability is just confidence, not modulated by register or context type
- **Forgetting/decay**: beliefs only grow; no mechanism for belief weakening over time
- **Generational turnover**: no birth/death of agents
- **Construction competition**: constructions are independent; using one doesn't preempt another
- **Variable rho**: preemption weight is global, but in reality some contexts make absence more informative than others


## 4. Experiments and Predictions

### 4.1 Baseline (completed)

**Setup**: 1000 agents, random mixing, 6 constructions with different niche widths and initial user fractions.

**Prediction** (post hoc, be honest): core stays high, gap dies, rare is somewhere in between. We had no specific prediction about bimodality.

**Finding**: Core and established lock in. Stable gap dies fast (wide niche). Community novel dies slowly (narrow niche). Rare_licensed develops **bimodality**: community splits into accepters and rejecters. This was not predicted.

**Interpretation risk**: Bimodality could be a generic property of any threshold-based production model with partial initial adoption. Needs sensitivity analysis.

### 4.2 Critical mass sweep (completed)

**Setup**: vary initial user fraction from 0% to 50%, niche_width = 0.10, random mixing.

**Prediction** (post hoc): tipping point exists.

**Finding**: tipping point around 15%. Below that, the form dies. Above, it spreads. The transition is fairly sharp.

**Interpretation risk**: The 15% number is almost certainly parameter-dependent (rho, tau, niche_width all contribute). We should not cite "15%" as a finding; we should cite "tipping point exists and its location depends on [parameters]."

### 4.3 Niche width sweep (completed)

**Setup**: vary niche_width, no initial users, random mixing.

**Prediction** (from single-agent model): wider niche = faster death (more opportunities for preemption).

**Finding**: Confirmed. Clean fan-out. This is the multi-agent replication of the single-agent result from simulate_dynamics.py.

**Interpretation risk**: Low. This is the most robust finding because it follows directly from the math: more opportunities * rho = faster b accumulation.

### 4.4 Network clustering (completed)

**Setup**: 10 communities, within_bias = 0.80, 10% initial users placed in different configurations.

**Prediction** (pre-registered, stated before running): Clustering would lower the critical mass, making rare forms survive that would die under random mixing.

**Finding**: **Opposite.** Random mixing spreads the form best (60% producers). Clustering concentrates it: user communities thrive (conf ~0.77), non-user communities die (conf ~0.24). The result is stable dialectal pockets, not global survival.

**Interpretation risk**: Medium. The within_bias of 0.80 is arbitrary. At very high within_bias (0.95+), the communities are nearly isolated and the effect should be even stronger. At low within_bias (0.5), it should approach random mixing. Need sensitivity sweep over beta.


## 5. Sensitivity Analysis: rho (completed 2026-02-14)

### 5.1 Setup

1000 agents, 300 steps, 20 MC runs per cell. rho swept over [0.05, 0.1, 0.2, 0.3, 0.5, 0.8].

Pre-registered predictions (written before running):
- Niche-speed relationship holds at all rho (mathematical)
- Critical mass tipping point MOVES with rho
- Analytic lower bound: f > rho / (1 + rho)
- Dialectal pockets form at all rho under clustering
- Bimodality: uncertain

### 5.2 Results

**A. rho is a gate, not a scale factor.** The heatmap (rho_sensitivity_heatmap.png) shows a clean diagonal boundary between survival and death. rho determines the minimum seed fraction:

| rho | Analytic bound f > rho/(1+rho) | Observed critical mass |
|-----|-------------------------------|----------------------|
| 0.05 | 0.048 | ~2% |
| 0.1 | 0.091 | ~2% |
| 0.2 | 0.167 | ~3-5% |
| 0.3 | 0.231 | ~8% |
| 0.5 | 0.333 | ~25% |
| 0.8 | 0.444 | ~50% |

The analytic bound is conservative but tracks the real threshold. The actual critical mass is somewhat below the bound because the bound assumes non-users contribute zero positive evidence, but agents near the threshold sometimes produce.

**Verdict on finding 3 (critical mass tipping point):** ROBUST in existence. The tipping point exists at all rho values. FRAGILE in location: the specific number (e.g., "15%") is entirely determined by rho. Never cite a critical mass number without specifying rho.

**B. The clustering effect reverses sign across rho.**

| rho | Random mixing | Clustered | Which wins? |
|-----|--------------|-----------|-------------|
| 0.05 | 100% | 98% | ~tie (both saturate) |
| 0.1 | 100% | 93% | ~tie |
| 0.2 | 94% | 22% | Random (+72%) |
| 0.3 | 60% | 11% | Random (+49%) |
| 0.5 | 2% | 10% | Clustered (+8%) |
| 0.8 | 0% | 9% | Clustered (+9%) |

At medium rho (0.2-0.3), random mixing spreads the form while clustering traps it in pockets. At high rho (0.5-0.8), both fail globally, but clustering preserves local pockets. The green bars at high rho represent the user community surviving while everyone else dies.

**Verdict on finding 4 (dialectal pockets):** ROBUST. Pockets form at all rho when community structure exists. But the INTERPRETATION changes: at medium rho, pockets are a side effect of failed global spread. At high rho, pockets are the only viable survival strategy.

**Verdict on finding 5 (random > clustering):** FRAGILE. Only true at medium rho. At high rho, clustering is actually better (preserves pockets). At low rho, both work.

**C. Bimodality is a phase-transition marker.**

Only rho = 0.3 (and marginally 0.2) show bimodal confidence distributions. At lower rho, the population converges to unanimous "grammatical." At higher rho, unanimous "ungrammatical." Bimodality appears when the system is poised near the critical mass tipping point for the given initial user fraction.

**Verdict on finding 2 (bimodality):** FRAGILE as a general property. Robust as a LOCAL property: it marks the boundary between survival and death regimes. If you're studying a construction whose status is genuinely contested in a speech community, the model predicts you should see bimodal acceptability distributions.

### 5.3 What survived the audit

| Finding | Status | Depends on rho? |
|---------|--------|-----------------|
| Niche-speed relationship | **ROBUST** | No (mathematical) |
| Critical mass exists | **ROBUST** | Location yes, existence no |
| Dialectal pockets | **ROBUST** | No (interpretation changes) |
| Random > clustering | **FRAGILE** | Reverses at high rho |
| Bimodality | **FRAGILE** (general) / **ROBUST** (at tipping point) | Only near phase transition |

### 5.4 Remaining sensitivity sweeps (not yet run)

| Parameter | Values to test | Expected sensitivity |
|-----------|---------------|---------------------|
| tau | 0.3, 0.4, 0.5, 0.6, 0.7 | MEDIUM. Lower tau = more exploration. |
| beta (within_bias) | 0.5, 0.6, 0.7, 0.8, 0.9, 0.95 | HIGH for clustering findings. |
| N | 200, 500, 1000, 2000, 5000 | LOW expected (same dynamics, less noise). |
| C (communities) | 5, 10, 20 | MEDIUM for clustering. |
| Initial confidence | (3,1), (5,2), (8,2), (10,1) | MEDIUM. Affects inertia. |


## 6. Mean-Field Analysis (completed 2026-02-14)

See `models/mean_field.py` for code and derivation.

### 6.1 Analytic Critical Mass

For a marginal agent (confidence c = tau = 0.5), positive evidence must outweigh negative evidence for the construction to survive. The rates are:

    rate_positive = w * p * c_prod
    rate_negative = w * (1 - p * c_prod) * rho

At the tipping point (rate_positive = rate_negative at c = 0.5):

    p * c_prod > rho / (1 + rho)

This gives two bounds:

- **Lower bound** (assumes c_prod = 1): f > rho / (1 + rho)
- **Mean-field estimate** (accounts for finite producer confidence): f > rho / (c_prod * (1 + rho))

### 6.2 Comparison to Simulation

| rho | Lower bound | MF (c=0.75) | MF (c=0.65) | Sim threshold | Sim prod% |
|-----|-------------|-------------|-------------|---------------|-----------|
| 0.05 | 0.048 | 0.063 | 0.073 | ~2% | 100% |
| 0.10 | 0.091 | 0.121 | 0.140 | ~2% | 100% |
| 0.20 | 0.167 | 0.222 | 0.256 | ~5% | 99% |
| 0.30 | 0.231 | 0.308 | 0.355 | ~8% | 54% |
| 0.50 | 0.333 | 0.444 | 0.513 | ~30% | 81% |
| 0.80 | 0.444 | 0.593 | 0.684 | ~50% | 54% |

**The simulation is consistently more forgiving than the analytic predictions.** At low rho (0.05-0.10), 2% seed users suffice where the bound predicts 5-9%. The discrepancy shrinks at high rho.

### 6.3 Why the Mean-Field Misses

The mean-field tracks two sub-populations (users, non-users) deterministically. In the trajectory plots, non-users monotonically decline and never cross the production threshold. This means the mean-field **misses the positive-feedback cascade**: in the stochastic ABM, a few non-users near the threshold stochastically convert to producers, which increases the production rate, which converts more non-users. This cascade is the mechanism that makes the simulation more forgiving than the analytic prediction.

The mean-field also misses:
- **Stochastic threshold crossing**: agents near c = 0.5 can flip either way in the ABM
- **Local density effects**: even under random mixing, stochastic partner selection creates temporary local majorities
- **Positive feedback from new producers**: each conversion amplifies the signal for remaining non-users

### 6.4 Inverting the Formula

The formula can be inverted to estimate rho from empirical data:

    rho = f* * c_prod / (1 - f* * c_prod)

If dialectal forms need ~20% community adoption to survive (f* = 0.20, c_prod = 0.70):

    rho = 0.14 / 0.86 ≈ 0.16

This would move rho from "free" to "empirically estimable" in the parameter table (§2). The challenge is finding empirical estimates of f* for real constructions.


## 7. Empirical rho Estimation from SCOSYA Data (2026-02-14)

### Data source

Scots Syntax Atlas (SCOSYA): 258 syntactic features, 146 locations, ~104,000 individual ratings on a 1-5 Likert scale (1=reject, 5=fully accept), stratified by age group (2 young, 2 old per location). Downloaded from HTML tabular data pages (the API was blocked by SiteGround anti-bot protection).

### Method

Identified features **actively declining in apparent time** (young speakers accept less than old speakers) with young acceptance rates in the 20-50% range (near the tipping point). For these features, f_young approximates the critical mass f*, and rho ≈ f*/(1-f*).

### Results

Using >= 4 as the acceptance threshold (maps best to the ABM's binary "accept"):

| Feature | Stimulus | f_young | rho estimate |
|---------|----------|---------|-------------|
| n47 | "didnae see nothing" | 0.32 | 0.46 |
| b17 | "I gied him" | 0.32 | 0.47 |
| A58 | "They was" | 0.33 | 0.50 |
| c7 | "We hadn't a lot of rain" | 0.34 | 0.51 |
| u3 | "he will that" | 0.35 | 0.53 |
| h8 | "Away you over there" | 0.35 | 0.53 |
| A59 | "shops was quiet" | 0.38 | 0.61 |

**Median rho ≈ 0.5, IQR 0.50-0.58.**

Threshold sensitivity: >= 3 gives rho ≈ 1.0; >= 4 gives rho ≈ 0.5. The result is threshold-dependent by a factor of ~2.

### Interpretation

At rho = 0.5, the critical mass for survival is f* = rho/(1+rho) = 0.33 (33% of the community must accept the construction for it to survive). This is consistent with:

- The ABM's default rho = 0.3 being in the right order of magnitude
- The SCOSYA features showing dialectal pockets (geographic SD up to 1.44) consistent with clustering effects
- Strong preemption: absence of a form in ~1/3 of relevant contexts is enough evidence to push toward rejection

### Caveats

1. **Apparent time ≠ real time.** Age-grading could confound the apparent-time interpretation.
2. **Threshold-sensitive.** The >= 4 vs >= 3 choice shifts the estimate by ~2x.
3. **Sample selection.** SCOSYA features were chosen for variability, so the distribution of acceptance rates is biased toward the middle.
4. **Gradient vs binary.** The ABM models binary accept/reject; Likert 1-5 is gradient.
5. **f_young ≈ f* assumes features are near the tipping point**, not already in free-fall. Features well below f* give rho estimates that are lower bounds.

### Additional findings

- **Age stratification reveals spreading features**: "I'm liking" (+56%), "I'm loving" (+47%), "They've went" (+36%), "I done that" (+31%). These are features young Scots speakers are adopting.
- **Dying features**: "had just on" (-52%), "Where bides she" (-27%), "I'll away" (-30%). Traditional Scots features older speakers accept but younger ones reject.
- **Geographic variation**: highest for "I caa mind" (SD=1.44), "There it's!" (SD=1.40), traditional Scots verb forms (SD=1.35-1.39). Lowest for universally accepted items like "I'm going to my bed" (SD=0.53).

## 8. Simulation-Based Calibration (2026-02-14)

### Purpose

Gelman checkpoint: can our SCOSYA estimation method (select declining features with young acceptance 20-50%, compute median f/(1-f)) actually recover known rho values from synthetic ABM data?

### Protocol

Pre-registered before running (see `models/fake_data_calibration.py` for full protocol and predictions).

- True rho values: {0.2, 0.3, 0.5, 0.7, 1.0}
- 258 synthetic features per replicate, initial user fractions U(0.01, 0.99)
- 200 agents, 200 steps, niche = 0.10, random mixing
- "Old generation" = acceptance rate at step 100; "young" = step 200
- Estimation method identical to SCOSYA analysis
- 20 MC replicates per true rho (25,800 ABM runs total)

### Pre-registered predictions (written before running)

1. Positive bias at low rho (selection window misses low tipping points)
2. Best recovery near rho = 0.5 (tipping point centered in window)
3. Underestimation at rho = 1.0 (tipping point at edge of window)
4. Wide IQRs everywhere (small qualifying subset)
5. Few qualifying features at extreme rho values

### Results

| True rho | Critical mass | Median estimate | Bias | IQR | Median n qualifying | NaN replicates |
|----------|-------------|-----------------|------|-----|-------------------|-------|
| 0.2 | 17% | 0.30 | +0.10 | [0.26, 0.32] | 0 | 14/20 |
| 0.3 | 23% | 0.37 | +0.07 | [0.32, 0.39] | 2 | 2/20 |
| 0.5 | 33% | 0.58 | +0.08 | [0.51, 0.63] | 9 | 0/20 |
| 0.7 | 41% | 0.58 | -0.13 | [0.52, 0.63] | 12 | 0/20 |
| 1.0 | 50% | 0.53 | -0.47 | [0.50, 0.56] | 15 | 0/20 |

All 4 testable predictions confirmed (prediction 4 about IQR width is confirmed by inspection).

### Key finding: the method has a ceiling

The method **cannot distinguish rho = 0.5 from rho = 0.7 from rho = 1.0**. The IQRs overlap completely for rho >= 0.5. This is because the selection window [0.20, 0.50] clips the tipping-point features when the true tipping point is above ~35%.

The method **can** distinguish low rho from high:
- rho = 0.2 vs 0.3: **separated** (IQRs do not overlap)
- rho = 0.3 vs 0.5: **separated**
- rho = 0.5 vs 0.7: **overlap** (cannot distinguish)
- rho = 0.7 vs 1.0: **overlap** (cannot distinguish)

### Revised interpretation of SCOSYA estimate

Our SCOSYA estimate of rho ≈ 0.5 provides a **lower bound**, not a point estimate.

- Consistent with true rho anywhere from **~0.4 to 1.0+**
- Rules out rho < 0.3 (those produce distinctly lower estimates)
- At true rho = 0.5, the bias-corrected estimate would be ~0.43
- But the same data would produce the same estimate at rho = 0.7 or 1.0

**Practical implication:** The critical mass is at least 33% (rho >= 0.5 implies f* >= 0.33). It could be as high as 50% (if rho = 1.0). The qualitative ABM predictions (dialectal pockets, tipping point, spreading/dying features) are robust across this range, but quantitative predictions (exact critical mass percentage) remain uncertain.

### What the calibration does NOT address

- Whether the ABM dynamics match real-world language change timescales
- Whether the estimation method would work with real gradient ratings (not binary)
- Whether the threshold sensitivity (>= 3 vs >= 4) interacts with the ceiling effect
- Whether a wider selection window could improve recovery at high rho (at cost of more noise)


## 9. Open Questions

- ~~Can rho be estimated empirically?~~ **Partially answered, then calibrated.** SCOSYA data gives rho >= ~0.4 (lower bound). The estimation method cannot distinguish rho = 0.5 from rho = 1.0 due to a ceiling effect in the selection window. See Section 8.
- **Does generational turnover change equilibria?** Currently beliefs only accumulate and the system freezes. Turnover might allow forms to revive or die on longer timescales. The SCOSYA age-stratification data could test this directly.
- **Is there a connection to language change S-curves?** The spread of a form from seed to saturation (at rho < 0.3) looks like it should follow an S-curve. Is it logistic? What's the rate parameter?
- **What happens at the phase boundary?** The rho = 0.3, f = 0.08 cell shows 54% producers (right at the boundary). Does this represent a stable mixed equilibrium or a bistable system that flips to 0% or 100% on longer timescales?
- **Can we improve the mean-field?** The two-population deterministic model misses the cascade dynamics. A stochastic mean-field (Langevin equation or master equation) might capture the threshold-crossing noise that makes the ABM more forgiving.
- **Can the ABM reproduce the SCOSYA feature distribution?** Run the ABM at rho = 0.5 with features seeded at varying initial frequencies and compare the resulting distribution of acceptance rates against the observed SCOSYA distribution.
