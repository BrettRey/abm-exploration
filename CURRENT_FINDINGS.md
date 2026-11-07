# Grammaticality population dynamics after the independent audits
<!-- SUMMARY: Current claim map for the February ABM and October mathematical investigations; consolidation complete, further research conditional on a paper claim or empirical test · status: parked · updated: 2026-11-07 -->

The model asks whether learning from use and non-use in relevant opportunities can sustain or extinguish a construction when speakers supply one another's evidence. It is an exploratory population extension of an earlier learning model in *Grammaticality de-idealized*. The October investigations establish qualified mathematical results for this specified model. They do not establish that the learning rule describes human communities.

This is the current entry point. `MODEL_SPEC.md`, `WRITEUP.md`, and the February log retain the historical claims, several of which were rejected. The audited code remains unchanged at source commit `e5d183e1b3642de7103b386c5413a9c51ec4242f`; the correction is to the interpretation and documentation. The reports linked below are preserved local investigation workspaces, not a publication or a new journal manuscript.

## Model and corrected results

An agent's confidence is the mean of a Beta evidence state. In a relevant encounter, hearing the construction adds one positive count; its absence adds a negative count weighted by rho. Production has a hard eligibility cutoff followed by a stochastic gate. The base model accumulates evidence without forgetting, generational turnover, analogy, or competition between explicitly represented constructions. Those choices constrain every conclusion below.

| Question | Checked conclusion | Limit |
|---|---|---|
| Is rho/(1+rho) a universal minimum seed fraction? | No. At cutoff one half, it is a critical heard-production frequency in a transformed evidence coordinate. | It is not an initial-user-fraction bound. |
| Must the production cutoff equal the non-user prior mean? | No. Counterexamples refute the claimed necessity and first-observation lockout. | The cutoff remains an independent model assumption. |
| Can a construction restart once every production gate closes? | No, in this model. Permanent silence is absorbing. | This relies on the absence of an innovation or other restart mechanism. |
| Is there a population-limit critical seed? | For the specified well-mixed model, original noisy prior, and rho = 0.3, the certified hard-cutoff interval is 7.14162–7.14168%. | Computer-assisted and model-specific; not an empirical estimate or a finite-population guarantee. |
| Does smoothing remove that threshold result? | The investigated smooth adoption rules retain critical seeds; the three certified widths raise their values. | This is a result for the specified smoothing family and priors, not arbitrary smooth responses. |
| What connects the limit to finite populations? | For the exact model and any fixed seed above the critical value, the probability that production probability tends to one approaches one as population size grows. | Conditional on the preceding population-limit theorem and its arithmetic contracts. Extinction remains possible at every finite size. |
| Does that make moderate populations reliable? | Reliability depends strongly on distance above the threshold. At N = 2,000 and 8% initial-user probability, 16/128 and 18/128 runs reached silence in the two studies. At 10%, both had 128/128 certified runs. | Certification bounds future failure by 0.1%; it does not mean a probability-one outcome. The two numerical-prior studies were not pooled. |
| Do network pockets persist? | Some normalized deterministic network equilibria are locally stable when the internal and external threshold conditions hold. | The original simulations do not establish universal or infinite-time stochastic pocket persistence. |
| Did SCOSYA calibrate a lower bound on rho? | No. The estimator's selection window and conflation of seed fraction, eligibility, and heard production invalidate that interpretation. | Existing numerical reproduction is not empirical identification. |

The six-cell finite-size study reproduced all 1,536 main trajectories. Its probability intervals concern the represented numerical priors under stated random-bit assumptions. Transfer to the ideal continuous Gaussian prior remains unproved. That is an optional follow-up if a particular numerical claim needs ideal-law probabilities.

## Relationship to the current OVMG paper

| Proposed use | Classification | Consequence for the manuscript |
|---|---|---|
| Use supplies evidence that affects later production | Applicable as a model-internal possibility | A bounded illustration of one feedback mechanism, with the learning and production assumptions stated. |
| Persistent absence is informative only in relevant opportunities | Applicable within the ABM's stipulated opportunity process | The opportunity measure and the informativity of omission still require independent linguistic evidence. |
| Finite-population persistence or the 7.14% threshold | Different assumptions | The current OVMG supplement has discounted evidence, a gated-choice likelihood, and a separate population transition. Do not transplant the ABM theorem or threshold into it. |
| The supplement's cusp, hysteresis, or derived obligatoriness | No direct validation | These have their own equations, code, and proof or verification obligations. The ABM audit does not settle them. |
| Structural coverage, compatibility, saturation, or operator-specific repair | No direct test | These distinctions are not represented by the one-construction ABM. |
| Grammaticality is an HPC kind, or community patterns constitute emergence | No entailment | A possible persistence mechanism does not establish the broader philosophical account. |

The ABM results remain a separate research record. Applying them to the OVMG formal supplement requires checking the assumptions for the particular claim.

## Next research decision

A candidate empirical target is **change in production or confidence following informative non-use, holding positive exposure and starting support comparable**. The comparison would pit opportunity-sensitive updating against an explicitly specified positive-evidence-only account. Opportunity status and alternative availability would need to be measured independently of the outcome. A contrast that merely compares frequent and rare forms would not identify the proposed mechanism.

**Bounded feasibility decision: park the empirical test with the current project materials.** The local `data/README.md` describes SCOSYA and MegaAcceptability rating resources; `MODEL_SPEC.md`, sections 7–8, describes a cross-sectional, age-stratified SCOSYA calibration. These descriptions don't specify a sequence connecting independently measured opportunities, informative non-use, positive exposure, and later within-speaker production or confidence. The age comparison isn't a longitudinal learning sequence. The existing calibration therefore doesn't identify the proposed contrast.

This is a finding about the documented project use of those resources, not a claim that no suitable dataset exists. The bounded search read `data/README.md` and searched `MODEL_SPEC.md` and `WRITEUP.md` for `scosya|data|confidence|before|after|opportunit|calibrat|longitudinal`, then checked the relevant calibration sections. No participant records were opened, no new data were downloaded, and no literature-wide search was attempted. Resume when a design or source supplies the missing exposure-to-outcome link. More repetitions of the old calibration wouldn't repair its estimand. No new participant study or collection protocol is approved here.

**Project status: parked.** Resume proof or simulation only for a named manuscript claim or a discriminating empirical test with the required exposure-to-outcome link.

## Evidence and reproduction records

- [Original audit and refutations](../abm-math-audit-20261027-27vg26cf/orchestration/reconciliation.md), including independent tests and the network result.
- [Certified threshold values](../abm-threshold-values-20261028-5kvt8u9w/orchestration/reconciliation.md), with exact endpoints and arithmetic qualifications.
- [Finite-population persistence theorem](../abm-finite-persistence-20261029-i0gvuq6v/orchestration/reconciliation.md) and [methods and scope](../abm-finite-persistence-20261029-i0gvuq6v/methods-and-scope.md).
- [Finite-size study](../abm-finite-size-20261029-pv6eijs7/orchestration/reconciliation.md), including sampler-specific intervals, stopping certificates, and excluded exploratory output.
- [Current OVMG source](../../papers/retarget/grammaticality-de-idealized/grammaticality-de-idealized.tex), [formal dynamics](../../papers/retarget/grammaticality-de-idealized/formal-dynamics.tex), and [repair decisions](../../papers/retarget/grammaticality-de-idealized/DECISIONS.md).
