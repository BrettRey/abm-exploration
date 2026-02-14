#!/usr/bin/env python3
"""
Sensitivity Sweeps for Remaining Parameters
=============================================

The rho sensitivity (MODEL_SPEC.md §5) showed rho controls everything.
This script audits the remaining free parameters at rho=0.5, our
empirical lower bound.

Parameters:
  A. tau (production threshold): 0.3, 0.4, 0.5, 0.6, 0.7
  B. beta (within_bias): 0.5, 0.6, 0.7, 0.8, 0.9, 0.95
  C. N (population size): 200, 500, 1000, 2000
  D. C_comm (communities): 5, 10, 20, 50
  E. Initial confidence: (3,1), (5,2), (8,2), (10,1)

All sweeps at rho=0.5 unless noted.

PRE-REGISTERED PREDICTIONS (written before running):
----------------------------------------------------

A. TAU (production threshold)
   - Lower tau -> lower effective critical mass. Agents produce more
     readily, so fewer initial users needed to sustain the form.
   - Higher tau -> higher critical mass. Agents hold back, fewer
     produce, more preemption.
   - Prediction: critical mass scales roughly with tau. At tau=0.3,
     the tipping point should be well below 33%. At tau=0.7, well above.
   - The analytic bound f > rho/(1+rho) assumes tau=0.5. It should
     underpredict at low tau and overpredict at high tau.

B. BETA (within_bias)
   - This is the clustering parameter. Higher beta = more insular
     communities.
   - Prediction: dialectal pockets get STRONGER with higher beta.
     At beta=0.95, communities are nearly isolated; user community
     should thrive, others die. At beta=0.5, approaches random mixing.
   - The sign-reversal found in the rho audit (random > clustered at
     medium rho) should hold at rho=0.5 for all beta values, since
     rho=0.5 is in the "high rho" regime where clustering preserves
     pockets but doesn't enable global spread.

C. N (population size)
   - Prediction: LOW sensitivity. Larger N should produce the same
     critical mass threshold but with less stochastic noise (tighter
     transitions). The dynamics are density-dependent, not size-dependent.
   - Possible exception: at small N (<200), finite-size effects might
     shift the threshold.

D. C_COMM (number of communities)
   - More communities = smaller communities = stronger isolation
     (each community has fewer members, so within-community
     interactions sample fewer people).
   - Prediction: more communities -> stronger dialectal pockets.
     But also: more communities means each one is smaller, so
     stochastic effects within communities are larger.
   - At C=50 with N=1000, each community has 20 agents. That's
     very small; expect high variance.

E. INITIAL CONFIDENCE
   - Higher initial confidence = more entrenched users. They
     take longer to be preempted, buying time for the cascade.
   - Prediction: higher confidence -> slightly lower critical mass
     (entrenched users resist preemption longer, giving marginal
     agents more positive evidence).
   - But the effect should be SMALL: confidence is a scale parameter
     on the Beta distribution, not a structural parameter.

Usage:
    python models/sensitivity_sweep.py
    python models/sensitivity_sweep.py --sweep tau
    python models/sensitivity_sweep.py --sweep beta
    python models/sensitivity_sweep.py --sweep all
"""

import random
import time
import argparse
from typing import List, Dict, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grammaticality_abm import Construction, Network, World, Agent, Belief

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ============================================================================
# CONFIGURATION
# ============================================================================

RHO = 0.5  # our empirical lower bound
DEFAULT_N = 1000
DEFAULT_STEPS = 300
DEFAULT_MC = 20
DEFAULT_NICHE = 0.10
DEFAULT_CLUSTER_NICHE = 0.04
DEFAULT_COMMUNITIES = 10
DEFAULT_WITHIN_BIAS = 0.80
DEFAULT_INIT_CONF = (6, 2)

FRACTIONS = [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]


# ============================================================================
# TAU SWEEP (requires modifying production threshold)
# ============================================================================

def make_tau_producer(tau):
    """Return a would_produce method with configurable threshold."""
    def would_produce(self, cxn_name):
        c = self.beliefs[cxn_name].confidence
        if c <= tau:
            return False
        return random.random() < c
    return would_produce


def sweep_tau(args):
    """Sweep production threshold tau."""
    tau_values = [0.3, 0.4, 0.5, 0.6, 0.7]
    n = args.agents
    steps = args.steps
    interact = n
    mc = args.mc_runs

    print("=" * 60)
    print(f"A. TAU SWEEP (production threshold) at rho={RHO}")
    print(f"  {n} agents, {steps} steps, {mc} MC runs")
    print(f"  tau values: {tau_values}")
    print("=" * 60)

    results = {}

    for tau in tau_values:
        results[tau] = {}
        print(f"\n  tau = {tau}")

        # Monkey-patch Agent.would_produce
        original_wp = Agent.would_produce
        Agent.would_produce = make_tau_producer(tau)

        for frac in FRACTIONS:
            prods = []
            for run in range(mc):
                cxns = [Construction("target", niche_width=DEFAULT_NICHE,
                                     initial_users=frac,
                                     initial_confidence=DEFAULT_INIT_CONF)]
                world = World(constructions=cxns, n_agents=n, rho=RHO,
                              random_seed=args.seed + run)
                world.run(n_steps=steps, interactions=interact)

                # Count producers using the SAME tau threshold
                prod = sum(1 for a in world.agents
                           if a.beliefs["target"].confidence > tau) / n
                prods.append(prod)

            mean_p = sum(prods) / len(prods)
            results[tau][frac] = mean_p
            survived = "ALIVE" if mean_p > 0.5 else "dead"
            print(f"    f={frac:>5.0%}  prod={mean_p:.0%}  [{survived}]")

        # Restore original
        Agent.would_produce = original_wp

    # Summary table
    print(f"\n  --- Critical mass thresholds ---")
    print(f"  {'tau':>6} {'Approx threshold':>18} {'Analytic bound':>16}")
    print(f"  {'-'*44}")
    for tau in tau_values:
        # Find approximate threshold (first frac with prod > 50%)
        threshold = ">50%"
        for frac in FRACTIONS:
            if results[tau][frac] > 0.5:
                threshold = f"~{frac:.0%}"
                break
        bound = RHO / (1 + RHO)
        print(f"  {tau:>6.1f} {threshold:>18} {bound:>16.3f}")

    if HAS_MPL and not args.no_viz:
        plot_sweep_heatmap(tau_values, FRACTIONS, results,
                           "tau", "Production threshold (tau)",
                           "tau_sensitivity.png")

    return results


# ============================================================================
# BETA (WITHIN_BIAS) SWEEP
# ============================================================================

def sweep_beta(args):
    """Sweep within-community bias."""
    beta_values = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    n = args.agents
    steps = args.steps
    interact = n
    mc = args.mc_runs

    print("\n" + "=" * 60)
    print(f"B. BETA SWEEP (within-community bias) at rho={RHO}")
    print(f"  {n} agents, {steps} steps, {mc} MC runs")
    print(f"  {DEFAULT_COMMUNITIES} communities, 10% initial users in community 0")
    print(f"  beta values: {beta_values}")
    print("=" * 60)

    results = {}

    for beta in beta_values:
        print(f"\n  beta = {beta}")

        # Two conditions: random mixing and clustered
        for cond_name, cond_cfg in [
            ("random", {"nc": 1, "wb": 0.0, "uc": None}),
            ("clustered", {"nc": DEFAULT_COMMUNITIES, "wb": beta, "uc": [0]}),
        ]:
            prods_all = []
            for run in range(mc):
                cxn = Construction("rare", niche_width=DEFAULT_CLUSTER_NICHE,
                                   initial_users=0.10,
                                   initial_confidence=(5, 2))
                net = Network(n, cond_cfg["nc"], cond_cfg["wb"])
                placement = {"rare": cond_cfg["uc"]} if cond_cfg["uc"] else {}

                world = World(constructions=[cxn], n_agents=n, rho=RHO,
                              network=net, user_placement=placement,
                              random_seed=args.seed + run)
                world.run(n_steps=steps, interactions=interact)

                prod = sum(1 for a in world.agents
                           if a.beliefs["rare"].confidence > 0.5) / n
                prods_all.append(prod)

                # Get community stats from last run
                if run == mc - 1 and cond_cfg["nc"] > 1:
                    comm = world.community_stats("rare")

            mean_p = sum(prods_all) / len(prods_all)
            results[(beta, cond_name)] = mean_p

            if cond_name == "clustered" and cond_cfg["nc"] > 1:
                user_comm = comm[0][1] if 0 in comm else 0
                other_comms = [comm[c][1] for c in comm if c != 0]
                other_mean = sum(other_comms) / len(other_comms) if other_comms else 0
                print(f"    {cond_name}: global={mean_p:.0%}  "
                      f"user_comm={user_comm:.0%}  others={other_mean:.0%}")
            else:
                print(f"    {cond_name}: global={mean_p:.0%}")

    # Summary
    print(f"\n  --- Clustering effect by beta ---")
    print(f"  {'beta':>6} {'Random':>8} {'Clustered':>10} {'Diff':>8}")
    print(f"  {'-'*36}")
    for beta in beta_values:
        r = results.get((beta, "random"), 0)
        c = results.get((beta, "clustered"), 0)
        print(f"  {beta:>6.2f} {r:>8.0%} {c:>10.0%} {r-c:>+8.0%}")

    return results


# ============================================================================
# N (POPULATION SIZE) SWEEP
# ============================================================================

def sweep_N(args):
    """Sweep population size."""
    N_values = [200, 500, 1000, 2000]
    steps = args.steps
    mc = min(args.mc_runs, 10)  # fewer MC for large N

    print("\n" + "=" * 60)
    print(f"C. N SWEEP (population size) at rho={RHO}")
    print(f"  {steps} steps, {mc} MC runs")
    print(f"  N values: {N_values}")
    print("=" * 60)

    results = {}

    for N in N_values:
        results[N] = {}
        interact = N
        t0 = time.time()
        print(f"\n  N = {N}")

        for frac in FRACTIONS:
            prods = []
            for run in range(mc):
                cxns = [Construction("target", niche_width=DEFAULT_NICHE,
                                     initial_users=frac,
                                     initial_confidence=DEFAULT_INIT_CONF)]
                world = World(constructions=cxns, n_agents=N, rho=RHO,
                              random_seed=args.seed + run)
                world.run(n_steps=steps, interactions=interact)

                prod = sum(1 for a in world.agents
                           if a.beliefs["target"].confidence > 0.5) / N
                prods.append(prod)

            mean_p = sum(prods) / len(prods)
            results[N][frac] = mean_p

        elapsed = time.time() - t0
        # Find threshold
        threshold = ">50%"
        for frac in FRACTIONS:
            if results[N][frac] > 0.5:
                threshold = f"~{frac:.0%}"
                break
        print(f"    threshold: {threshold}  [{elapsed:.1f}s]")

    # Summary
    print(f"\n  --- Critical mass by N ---")
    print(f"  {'N':>6}", end="")
    for frac in FRACTIONS:
        print(f"  {frac:>5.0%}", end="")
    print()
    print(f"  {'-'*62}")
    for N in N_values:
        print(f"  {N:>6}", end="")
        for frac in FRACTIONS:
            p = results[N][frac]
            print(f"  {p:>5.0%}", end="")
        print()

    return results


# ============================================================================
# C (COMMUNITIES) SWEEP
# ============================================================================

def sweep_C(args):
    """Sweep number of communities."""
    C_values = [5, 10, 20, 50]
    n = args.agents
    steps = args.steps
    interact = n
    mc = args.mc_runs

    print("\n" + "=" * 60)
    print(f"D. C SWEEP (number of communities) at rho={RHO}")
    print(f"  {n} agents, beta={DEFAULT_WITHIN_BIAS}, 10% initial users in comm 0")
    print(f"  C values: {C_values}")
    print("=" * 60)

    results = {}

    for C in C_values:
        print(f"\n  C = {C} (community size = {n // C})")

        prods_all = []
        for run in range(mc):
            cxn = Construction("rare", niche_width=DEFAULT_CLUSTER_NICHE,
                               initial_users=0.10,
                               initial_confidence=(5, 2))
            net = Network(n, C, DEFAULT_WITHIN_BIAS)
            placement = {"rare": [0]}

            world = World(constructions=[cxn], n_agents=n, rho=RHO,
                          network=net, user_placement=placement,
                          random_seed=args.seed + run)
            world.run(n_steps=steps, interactions=interact)

            prod = sum(1 for a in world.agents
                       if a.beliefs["rare"].confidence > 0.5) / n
            prods_all.append(prod)

        comm = world.community_stats("rare")
        user_comm = comm[0][1] if 0 in comm else 0
        other_comms = [comm[c][1] for c in comm if c != 0]
        other_mean = sum(other_comms) / len(other_comms) if other_comms else 0

        mean_p = sum(prods_all) / len(prods_all)
        results[C] = {
            "global": mean_p,
            "user_comm": user_comm,
            "other_mean": other_mean,
        }
        print(f"    global={mean_p:.0%}  "
              f"user_comm={user_comm:.0%}  others={other_mean:.0%}")

    return results


# ============================================================================
# INITIAL CONFIDENCE SWEEP
# ============================================================================

def sweep_init_conf(args):
    """Sweep initial user confidence."""
    conf_values = [(3, 1), (5, 2), (8, 2), (10, 1)]
    n = args.agents
    steps = args.steps
    interact = n
    mc = args.mc_runs

    print("\n" + "=" * 60)
    print(f"E. INITIAL CONFIDENCE SWEEP at rho={RHO}")
    print(f"  {n} agents, {steps} steps, {mc} MC runs")
    print(f"  Confidence values: {conf_values}")
    print("=" * 60)

    results = {}

    for a_init, b_init in conf_values:
        conf_key = (a_init, b_init)
        results[conf_key] = {}
        mean_conf = a_init / (a_init + b_init)
        print(f"\n  Beta({a_init}, {b_init}) -> mean confidence = {mean_conf:.2f}")

        for frac in FRACTIONS:
            prods = []
            for run in range(mc):
                cxns = [Construction("target", niche_width=DEFAULT_NICHE,
                                     initial_users=frac,
                                     initial_confidence=(a_init, b_init))]
                world = World(constructions=cxns, n_agents=n, rho=RHO,
                              random_seed=args.seed + run)
                world.run(n_steps=steps, interactions=interact)

                prod = sum(1 for a in world.agents
                           if a.beliefs["target"].confidence > 0.5) / n
                prods.append(prod)

            mean_p = sum(prods) / len(prods)
            results[conf_key][frac] = mean_p

        threshold = ">50%"
        for frac in FRACTIONS:
            if results[conf_key][frac] > 0.5:
                threshold = f"~{frac:.0%}"
                break
        print(f"    threshold: {threshold}")

    return results


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_sweep_heatmap(param_values, fractions, results,
                       param_name, param_label, filename):
    """Generic heatmap for parameter × initial_users → producer fraction."""
    fig, ax = plt.subplots(figsize=(11, 5))

    mat = []
    for pv in param_values:
        row = [results[pv][f] for f in fractions]
        mat.append(row)
    mat = np.array(mat)

    im = ax.imshow(mat, aspect='auto', origin='lower', cmap='RdYlGn',
                   vmin=0, vmax=1,
                   extent=[-0.5, len(fractions) - 0.5,
                           -0.5, len(param_values) - 0.5])

    for i, pv in enumerate(param_values):
        for j, frac in enumerate(fractions):
            val = mat[i, j]
            color = "white" if val < 0.3 or val > 0.7 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold")

    ax.set_xticks(range(len(fractions)))
    ax.set_xticklabels([f"{f:.0%}" for f in fractions])
    ax.set_yticks(range(len(param_values)))
    ax.set_yticklabels([str(v) for v in param_values])
    ax.set_xlabel("Initial user fraction", fontsize=11)
    ax.set_ylabel(param_label, fontsize=11)
    ax.set_title(f"Critical Mass × {param_label} (rho={RHO})", fontsize=12)

    fig.colorbar(im, ax=ax, label="Final producer fraction", shrink=0.8)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"\n  Plot saved: {filename}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sensitivity sweeps for remaining parameters")
    parser.add_argument("--sweep", default="all",
                        help="Which sweep: tau, beta, N, C, conf, all")
    parser.add_argument("--agents", type=int, default=DEFAULT_N)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--mc-runs", type=int, default=DEFAULT_MC)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    t_total = time.time()

    all_results = {}

    if args.sweep in ("tau", "all"):
        all_results["tau"] = sweep_tau(args)
    if args.sweep in ("beta", "all"):
        all_results["beta"] = sweep_beta(args)
    if args.sweep in ("N", "all"):
        all_results["N"] = sweep_N(args)
    if args.sweep in ("C", "all"):
        all_results["C"] = sweep_C(args)
    if args.sweep in ("conf", "all"):
        all_results["conf"] = sweep_init_conf(args)

    print(f"\n{'='*60}")
    print(f"Total wall time: {time.time() - t_total:.1f}s")
    print(f"{'='*60}")

    return all_results


if __name__ == "__main__":
    main()
