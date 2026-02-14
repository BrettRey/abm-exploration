# Population Dynamics of Grammaticality: An ABM Exploration

## Summary

We built a multi-agent extension of the OVMG's Bayesian evidence-accumulation model of grammaticality. The single-agent model (Reynolds, in prep) shows how preemption from absence-in-relevant-contexts can push a construction toward rejection. The multi-agent version makes preemption *endogenous*: an agent's absence-of-evidence depends on what its neighbours actually produce.

The model has one genuinely free parameter: rho, the preemption weight. Everything else is either theory-fixed, empirically estimable, or structurally constrained.

## What the model does

1,000 agents hold Beta-distributed beliefs about constructions. At each time step, pairs interact: a speaker either produces a construction (positive evidence, a += 1) or doesn't (negative evidence, b += rho). An agent produces if its confidence exceeds 0.5 and a stochastic gate fires. Agents live in communities with biased within-community interaction.

## Key findings

### 1. Critical mass is rho/(1+rho)

For a construction to survive, it needs a minimum fraction of initial users. The analytic lower bound is f* > rho/(1+rho). At rho = 0.5, this is 33%. The ABM confirms the bound and shows the simulation is slightly more forgiving (stochastic threshold-crossing creates a positive-feedback cascade the mean-field misses).

### 2. Network clustering creates dialectal pockets, not global survival

Concentrating initial users in one community produces stable pockets: the user community thrives while others die. This contradicts our pre-registered prediction that clustering would lower the critical mass. The effect reverses with rho: at medium rho, random mixing is better for global spread; at high rho, clustering is the only survival strategy.

### 3. rho >= 0.4 from SCOSYA dialect data

We estimated rho from the Scots Syntax Atlas (258 features, 146 locations, 104K ratings). Features declining in apparent time with young acceptance rates near the tipping point give rho ≈ 0.5.

Simulation-based calibration (25,800 ABM runs, pre-registered predictions) showed this is a **lower bound**, not a point estimate. The estimation method has a fundamental ceiling: it cannot distinguish rho = 0.5 from rho = 0.7 from rho = 1.0, because the ABM dynamics push features away from the tipping point faster than they can be observed. Window widening does not help. The method rules out rho < 0.3.

### 4. tau = E[non-user prior] is a derived constraint

The sensitivity audit found a phase transition at the production threshold tau = 0.5, which equals the non-user prior mean (Beta(1,1)). We derived this analytically:

- Below: non-users produce immediately (trivial survival)
- Above: non-users never start producing (trivial death)
- At equality: the first interaction determines each agent's trajectory, and population composition determines outcomes (critical mass)

This is not a fragility. It is the model correctly identifying that population dynamics only matter for constructions where speakers are genuinely uncertain. The constraint tau = E[prior] is structurally necessary, reducing the effective free parameter count to one (rho).

Verified across four different priors (Beta(1,1), Beta(2,1), Beta(1,2), Beta(3,3)).

### 5. Empirical validation against SCOSYA

Four tests of ABM predictions against SCOSYA patterns, all supported:

1. **Critical mass**: 63% of features below the 33% threshold are declining vs 29% above
2. **Dialectal pockets**: geographic SD peaks for contested features (0.94 vs 0.63)
3. **Bimodality**: features below threshold have more "rejected" ratings (48% vs 21%)
4. **Spreading**: all 10 fastest-spreading features have acceptance above 33%

These tests are qualitative and post-hoc. The simulation-based calibration was the first properly pre-registered analysis.

## Sensitivity audit summary

| Parameter | Sensitivity | Finding |
|-----------|-------------|---------|
| rho | **CRITICAL** | Gates everything. The only free parameter that matters. |
| tau | **DERIVED** | Phase transition at tau = E[prior]. Structurally constrained. |
| beta (within_bias) | LOW | Pockets form at all values. |
| N (population) | LOW | Same threshold at all N. |
| C (communities) | MODERATE | More communities, sharper pockets. |
| Initial confidence | MODERATE | More entrenched, lower critical mass. |

## Methodological contributions

1. **Calibrate on fake data before trusting real data.** The SBC turned a claimed point estimate into an honest lower bound. The result is weaker but trustworthy.

2. **Derive your sensitivities.** The tau phase transition looked like a fatal fragility until we traced the math. The derivation transformed "model only works at one setting" into "model correctly constrains to the interesting regime."

3. **Pre-register before running.** Only the clustering experiment (wrong) and the SBC (confirmed) had genuine pre-registered predictions. Everything else was post-hoc. Gelman would approve of the SBC; the rest is night science.

## What this means for the grammaticality paper

The ABM provides a computational existence proof: if you take the OVMG's evidence-accumulation framework seriously and make it multi-agent, you get critical mass phenomena, dialectal pockets, and a principled distinction between rare-but-licensed and accidental gaps. These are emergent properties, not built in.

The calibration story (rho >= 0.4 from SCOSYA, ceiling is fundamental, tau is derived) shows the model is not a pure intuition pump: it makes constrained predictions even though the key parameter cannot be precisely estimated.

The model has clear limitations: no generational turnover, no analogical support, no construction competition, no register variation, no metalinguistic knowledge. But it's the simplest model that produces the target phenomena from OVMG commitments, and its one free parameter has an empirical lower bound.

## Files

| File | Description |
|------|-------------|
| `models/grammaticality_abm.py` | Main ABM (6 experiments) |
| `models/sensitivity_sweep.py` | tau, beta, N, C, init_conf sweeps |
| `models/fake_data_calibration.py` | Simulation-based calibration |
| `models/window_sweep_calibration.py` | Window sweep (ceiling confirmation) |
| `models/tau_derivation.py` | Analytic derivation of tau constraint |
| `models/mean_field.py` | Mean-field analysis |
| `MODEL_SPEC.md` | Formal specification with full results |
| `PROJECT_LOG.md` | Chronological exploration log |
| `data/scots/` | SCOSYA data and download scripts |
