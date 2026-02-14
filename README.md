# ABM Exploration

Exploring agent-based modelling as a method, with linguistics as the application domain.

**Status:** Night science. No commitments. First model built and audited (2026-02-14).

**Prior art:** `papers/HPC book/code/countability_abm.py` (didactic mechanism sketch for countability, Ch 9). Pattern: agents with lexicons, constructions as "locks," bidirectional inference, multi-timescale maintenance, ablation stress tests.

## What exists

### Grammaticality Population Dynamics ABM

`models/grammaticality_abm.py` (~900 lines, raw Python + matplotlib)

Takes the single-agent Bayesian evidence-accumulation model from `papers/Grammaticality_de_idealized/simulate_dynamics.py` and makes it multi-agent. Key move: preemption (negative evidence from not hearing a form) becomes endogenous rather than stipulated.

```bash
python models/grammaticality_abm.py                 # baseline (1000 agents)
python models/grammaticality_abm.py --exp=cluster    # network clustering experiment
python models/grammaticality_abm.py --exp=rho        # rho sensitivity audit
python models/grammaticality_abm.py --exp=all        # everything (~2.5 min)
```

**Findings (with caveats):**
1. Rare-vs-gap distinction emerges endogenously (niche width controls death speed)
2. Network clustering creates dialectal pockets, not global survival
3. rho (preemption weight) is the key free parameter; it gates everything
4. Bimodality appears only near the phase transition (contested constructions)

See `MODEL_SPEC.md` for the formal model, parameter classifications, and sensitivity analysis.

### Answered open questions

- **Mesa or raw Python?** Raw Python. Fast, transparent, no framework overhead.
- **Notebooks or scripts?** Scripts with matplotlib. Easy to run, reproducible.
- **Simplest model that teaches something?** Beta-distributed beliefs with endogenous preemption. ~100 lines of core model, the rest is experiments and plotting.

## Modelable claims from the deitality paper

The paper (Reynolds, "Definiteness and Deitality in English") makes several causal claims that are ABM-susceptible:

### 1. Grammaticalization pathway (diachronic mechanism)
Demonstratives grammaticalize into definite articles, "dragging" distributional properties with them. An ABM could show:
- Agents using deictic forms in context
- Semantic bleaching over generations (deictic to general identifiability)
- Distributional properties persisting even as semantics shifts
- **Prediction:** The property cluster should remain stable through the transition

### 2. Acquisition and prototype formation (acquisitional mechanism)
Children overgeneralize `the` before learning subtle conditions. An ABM could show:
- Learner agents exposed to frequency-skewed input
- Prototype formation around the most frequent exponent
- Overgeneralization followed by gradual refinement
- **Prediction:** The acquisition pathway should reproduce the overgeneralization stage

### 3. Property cluster maintenance (synchronic mechanism)
The 6 deitality properties co-occur because of syntactic feature-checking. An ABM could show:
- How formal constraints enforce co-occurrence
- What happens when you knock out individual properties (ablation)
- **Prediction:** Removing one mechanism should degrade cluster coherence but not destroy it (HPC resilience)

### 4. Weak definite emergence
How `go to the hospital` becomes a stable construction type with its own distributional profile. An ABM could show:
- Speakers conventionalizing V+the+N patterns
- Lexical restrictions emerging from frequency and functional specialization
- **Prediction:** Weak definites should emerge as a stable sub-region of the deitality space

### 5. Cross-linguistic variation
Why some languages grammaticalize definiteness morphologically and others don't. An ABM could explore:
- Different initial conditions (demonstrative inventory, word order)
- Which conditions produce article systems vs. bare-nominal systems
- **Prediction:** Article emergence should correlate with specific communicative-pressure configurations

## Other ABM-susceptible ideas from the portfolio

Surveyed 2026-02-14. See the explore agent's report for details.

- **Constructions as mesoscales** (CE 2.0): agents converging on mesoscale granularity through interaction
- **CxG + chunking + feature selection**: compression-under-noise selects which features constructions monitor
- **Varieties as conditioning structure**: population ABM with agents conditioning on different variables
- **Grammar and emergence** (with Nefdt): operators as convergent functional solutions, bottom-up norm enforcement
