#!/usr/bin/env python3
"""
Grammaticality Population Dynamics ABM
=======================================

Night science: what happens when you make the OVMG's Bayesian evidence-
accumulation model multi-agent?

Key move: in simulate_dynamics.py, opportunity rate is EXOGENOUS (you set
n_opportunities_per_step). Here, preemption becomes ENDOGENOUS: you accumulate
negative evidence by not hearing other speakers use a form in contexts where
it could have appeared.

The rare-vs-gap distinction should EMERGE from population dynamics.

v2: Network structure. Agents live in communities. Within-community interaction
is more frequent than between-community. Does clustering save rare forms?

Based on:
  - simulate_dynamics.py (single-agent Bayesian trajectories)
  - countability_abm.py (multi-agent ABM pattern)

Usage:
    python models/grammaticality_abm.py                     # baseline (1000 agents)
    python models/grammaticality_abm.py --exp=sweep         # critical-mass sweep
    python models/grammaticality_abm.py --exp=niche         # niche-width sweep
    python models/grammaticality_abm.py --exp=cluster       # network clustering experiment
    python models/grammaticality_abm.py --exp=all           # everything
"""

import random
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import argparse

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Note: matplotlib not available; text output only")


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Construction:
    """A grammatical construction with a communicative niche."""
    name: str
    niche_width: float       # P(random context is relevant)
    initial_users: float     # fraction of agents who start confident
    initial_confidence: Tuple[float, float] = (8.0, 2.0)  # Beta(a,b) for users
    description: str = ""


@dataclass
class Belief:
    """Beta-distributed belief about whether a construction is grammatical."""
    __slots__ = ('a', 'b')
    a: float
    b: float

    @property
    def confidence(self) -> float:
        return self.a / (self.a + self.b)


# ============================================================================
# NETWORK
# ============================================================================

class Network:
    """
    Community-structured social network.

    Agents belong to communities. Interactions are biased toward within-community
    pairs (controlled by within_bias). Setting n_communities=1 or within_bias=0.0
    gives random mixing (equivalent to v1).
    """

    def __init__(self, n_agents: int, n_communities: int = 1,
                 within_bias: float = 0.8):
        self.n_agents = n_agents
        self.n_communities = n_communities
        self.within_bias = within_bias

        # assign agents round-robin to communities
        self.membership = [i % n_communities for i in range(n_agents)]
        self.communities: Dict[int, List[int]] = defaultdict(list)
        for i, c in enumerate(self.membership):
            self.communities[c].append(i)

    def sample_pair(self) -> Tuple[int, int]:
        """Return (speaker_idx, hearer_idx)."""
        speaker = random.randrange(self.n_agents)
        if (self.n_communities > 1
                and random.random() < self.within_bias):
            # within-community
            members = self.communities[self.membership[speaker]]
            if len(members) < 2:
                hearer = random.randrange(self.n_agents)
                while hearer == speaker:
                    hearer = random.randrange(self.n_agents)
            else:
                hearer = speaker
                while hearer == speaker:
                    hearer = random.choice(members)
        else:
            # random cross-community
            hearer = random.randrange(self.n_agents)
            while hearer == speaker:
                hearer = random.randrange(self.n_agents)
        return speaker, hearer


# ============================================================================
# AGENT
# ============================================================================

class Agent:
    __slots__ = ('id', 'beliefs')

    def __init__(self, agent_id: int, beliefs: Dict[str, Belief]):
        self.id = agent_id
        self.beliefs = beliefs

    def would_produce(self, cxn_name: str) -> bool:
        c = self.beliefs[cxn_name].confidence
        if c <= 0.5:
            return False
        return random.random() < c

    def hear_production(self, cxn_name: str):
        self.beliefs[cxn_name].a += 1.0

    def hear_absence(self, cxn_name: str, rho: float):
        self.beliefs[cxn_name].b += rho


# ============================================================================
# WORLD
# ============================================================================

class World:
    def __init__(
        self,
        constructions: Optional[List[Construction]] = None,
        n_agents: int = 1000,
        rho: float = 0.3,
        network: Optional[Network] = None,
        user_placement: Optional[Dict[str, List[int]]] = None,
        random_seed: Optional[int] = None,
    ):
        """
        Args:
            constructions: list of Construction specs (default: the standard 6)
            n_agents: population size
            rho: preemption weight (how much absence-of-evidence counts)
            network: Network object for structured interaction (default: random mixing)
            user_placement: {cxn_name: [community_ids]} to concentrate initial
                           users in specific communities. If not given, users
                           are scattered randomly.
            random_seed: for reproducibility
        """
        if random_seed is not None:
            random.seed(random_seed)

        self.rho = rho
        self.n_agents = n_agents
        self.time_step = 0
        self.history: List[Dict] = []

        if constructions is None:
            constructions = default_constructions()
        self.constructions = {c.name: c for c in constructions}

        if network is None:
            network = Network(n_agents)
        self.network = network
        self.user_placement = user_placement or {}

        self.agents = self._make_agents(n_agents)

        # cache construction list for hot loop
        self._cxn_items = list(self.constructions.items())

    def _make_agents(self, n: int) -> List[Agent]:
        agents = []
        for i in range(n):
            beliefs = {}
            for name, cxn in self.constructions.items():
                is_user = False
                if name in self.user_placement:
                    # place users only in specified communities
                    target_comms = self.user_placement[name]
                    if self.network.membership[i] in target_comms:
                        # scale up probability so total users ≈ initial_users * n
                        n_target = sum(len(self.network.communities[c])
                                       for c in target_comms)
                        target_count = cxn.initial_users * n
                        prob = min(1.0, target_count / max(1, n_target))
                        is_user = random.random() < prob
                else:
                    is_user = random.random() < cxn.initial_users

                if is_user:
                    a, b = cxn.initial_confidence
                    a += random.gauss(0, 0.5)
                    b += abs(random.gauss(0, 0.3))
                    beliefs[name] = Belief(a=max(1.0, a), b=max(0.5, b))
                else:
                    beliefs[name] = Belief(a=1.0, b=1.0)
            agents.append(Agent(agent_id=i, beliefs=beliefs))
        return agents

    def step(self, interactions: int):
        self.time_step += 1
        rho = self.rho
        cxn_items = self._cxn_items
        agents = self.agents
        net = self.network
        _random = random.random

        for _ in range(interactions):
            si, hi = net.sample_pair()
            speaker = agents[si]
            hearer = agents[hi]

            for name, cxn in cxn_items:
                if _random() < cxn.niche_width:
                    if speaker.would_produce(name):
                        hearer.beliefs[name].a += 1.0
                    else:
                        hearer.beliefs[name].b += rho

    def run(self, n_steps: int, interactions: int):
        for _ in range(n_steps):
            self.step(interactions)

    def record_snapshot(self):
        """Take a snapshot of current state and append to history."""
        stats = {"time": self.time_step}
        for name in self.constructions:
            confs = [a.beliefs[name].confidence for a in self.agents]
            n = len(confs)
            mean = sum(confs) / n
            std = (sum((c - mean) ** 2 for c in confs) / n) ** 0.5
            producers = sum(1 for c in confs if c > 0.5) / n
            stats[f"{name}_mean"] = mean
            stats[f"{name}_std"] = std
            stats[f"{name}_prod"] = producers
        self.history.append(stats)

    def run_and_record(self, n_steps: int, interactions: int,
                       record_every: int = 1):
        """Run simulation, recording snapshots at intervals."""
        for i in range(n_steps):
            self.step(interactions)
            if (i + 1) % record_every == 0:
                self.record_snapshot()

    def community_stats(self, cxn_name: str) -> Dict[int, Tuple[float, float]]:
        """Per-community mean confidence and producer fraction."""
        result = {}
        for comm_id, members in self.network.communities.items():
            confs = [self.agents[m].beliefs[cxn_name].confidence for m in members]
            mean = sum(confs) / len(confs)
            prod = sum(1 for c in confs if c > 0.5) / len(confs)
            result[comm_id] = (mean, prod)
        return result

    def print_status(self, label: str = ""):
        if label:
            print(f"\n{label} (t={self.time_step})")
        else:
            print(f"\nt={self.time_step}")
        print(f"  {'Construction':<20} {'Confidence':>10} {'  Std':>7} {'Producers':>10}")
        print(f"  {'-'*50}")
        for name in self.constructions:
            confs = [a.beliefs[name].confidence for a in self.agents]
            n = len(confs)
            m = sum(confs) / n
            s = (sum((c - m) ** 2 for c in confs) / n) ** 0.5
            p = sum(1 for c in confs if c > 0.5) / n
            print(f"  {name:<20} {m:>10.3f} {s:>7.3f} {p:>9.0%}")


# ============================================================================
# DEFAULT CONSTRUCTIONS
# ============================================================================

def default_constructions() -> List[Construction]:
    return [
        Construction(
            "core", niche_width=0.30, initial_users=0.95,
            initial_confidence=(10, 1),
            description="Fully grammatical, wide niche (basic SVO)"),
        Construction(
            "established", niche_width=0.15, initial_users=0.80,
            initial_confidence=(7, 2),
            description="Grammatical, moderate niche (topicalization)"),
        Construction(
            "rare_licensed", niche_width=0.04, initial_users=0.10,
            initial_confidence=(5, 2),
            description="Rare but licensed, narrow niche (friend of whose)"),
        Construction(
            "stable_gap", niche_width=0.25, initial_users=0.0,
            initial_confidence=(8, 2),
            description="Ungrammatical, wide niche (*Who did you wonder whether left?)"),
        Construction(
            "community_novel", niche_width=0.02, initial_users=0.0,
            initial_confidence=(8, 2),
            description="Unseen, very narrow niche (novel blend)"),
        Construction(
            "emerging", niche_width=0.10, initial_users=0.05,
            initial_confidence=(6, 2),
            description="Small seed, moderate niche (tipping point?)"),
    ]


# ============================================================================
# EXPERIMENTS
# ============================================================================

def experiment_baseline(args):
    """Scaled-up baseline: watch the dynamics with 1000+ agents."""
    n = args.agents
    steps = args.steps
    interact = args.interactions or n  # default: interactions = n_agents
    record_every = max(1, steps // 100)  # ~100 data points

    print("=" * 60)
    print("BASELINE: Population grammaticality dynamics")
    print(f"  {n} agents, {steps} steps, {interact} interactions/step, rho={args.rho}")
    print("=" * 60)

    world = World(n_agents=n, rho=args.rho, random_seed=args.seed)
    world.record_snapshot()  # t=0
    world.print_status("Initial")

    t0 = time.time()
    world.run_and_record(steps, interact, record_every)
    elapsed = time.time() - t0

    world.print_status("Final")
    print(f"\n  Elapsed: {elapsed:.1f}s "
          f"({steps * interact / 1e6:.1f}M interactions)")

    if HAS_MPL and not args.no_viz:
        plot_trajectories(world, "grammaticality_baseline.png")
        plot_distributions(world, "confidence_distributions.png")

    return world


def experiment_critical_mass(args):
    """Sweep initial_users fraction to find tipping point."""
    n = args.agents
    steps = args.steps
    interact = args.interactions or n

    print("=" * 60)
    print("CRITICAL MASS SWEEP")
    print(f"  {n} agents, {steps} steps, niche_width=0.10")
    print("=" * 60)

    fractions = [0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.50]
    results = {}

    for frac in fractions:
        cxns = [
            Construction("target", niche_width=0.10, initial_users=frac,
                         initial_confidence=(6, 2)),
            Construction("core", niche_width=0.30, initial_users=0.95,
                         initial_confidence=(10, 1)),
        ]
        world = World(constructions=cxns, n_agents=n,
                       rho=args.rho, random_seed=args.seed)
        world.run(n_steps=steps, interactions=interact)
        world.record_snapshot()

        m = world.history[-1]["target_mean"]
        p = world.history[-1]["target_prod"]
        results[frac] = (m, p)
        print(f"  initial_users={frac:>5.0%}  =>  "
              f"conf={m:.3f}  producers={p:.0%}")

    if HAS_MPL and not args.no_viz:
        plot_critical_mass(fractions, results)

    return results


def experiment_niche_sweep(args):
    """Sweep niche_width: wide niche = fast death, narrow = slow."""
    n = args.agents
    steps = args.steps
    interact = args.interactions or n
    record_every = max(1, steps // 100)

    print("=" * 60)
    print("NICHE WIDTH SWEEP")
    print(f"  {n} agents, {steps} steps, no initial users")
    print("=" * 60)

    widths = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40]
    all_histories = {}

    for w in widths:
        cxns = [Construction("target", niche_width=w, initial_users=0.0,
                             initial_confidence=(8, 2))]
        world = World(constructions=cxns, n_agents=n,
                       rho=args.rho, random_seed=args.seed)
        world.run_and_record(steps, interact, record_every)

        traj = [s["target_mean"] for s in world.history]
        all_histories[w] = traj
        print(f"  niche={w:.3f}  =>  final conf={traj[-1]:.3f}")

    if HAS_MPL and not args.no_viz:
        plot_niche_sweep(all_histories)

    return all_histories


def experiment_clustering(args):
    """
    THE MAIN EVENT: does network clustering save rare forms?

    Four conditions, same total fraction of initial users (10%):
      1. random_mixing  - no community structure
      2. scattered      - 10 communities, users spread evenly
      3. clustered_2    - 10 communities, users in 2 communities
      4. clustered_1    - 10 communities, users all in 1 community

    Monte Carlo: run each condition multiple times for confidence bands.
    """
    n = args.agents
    steps = args.steps
    interact = args.interactions or n
    record_every = max(1, steps // 100)
    n_runs = args.mc_runs
    n_communities = 10
    within_bias = 0.80
    user_frac = 0.10  # 10% initial users, same across conditions
    niche = 0.04      # narrow niche (the "friend of whose" regime)

    print("=" * 60)
    print("NETWORK CLUSTERING EXPERIMENT")
    print(f"  {n} agents, {n_communities} communities, "
          f"within_bias={within_bias}")
    print(f"  {user_frac:.0%} initial users, niche_width={niche}")
    print(f"  {steps} steps, {interact} interactions/step")
    print(f"  {n_runs} Monte Carlo runs per condition")
    print("=" * 60)

    conditions = {
        "random_mixing": {
            "n_communities": 1,
            "within_bias": 0.0,
            "user_communities": None,
        },
        "scattered": {
            "n_communities": n_communities,
            "within_bias": within_bias,
            "user_communities": None,  # spread randomly
        },
        "clustered_2": {
            "n_communities": n_communities,
            "within_bias": within_bias,
            "user_communities": [0, 1],
        },
        "clustered_1": {
            "n_communities": n_communities,
            "within_bias": within_bias,
            "user_communities": [0],
        },
    }

    all_results = {}  # condition -> list of trajectory arrays

    for cond_name, cfg in conditions.items():
        print(f"\n  --- {cond_name} ---")
        trajectories = []
        prod_trajectories = []

        for run in range(n_runs):
            seed = args.seed + run
            cxn = Construction("rare", niche_width=niche,
                               initial_users=user_frac,
                               initial_confidence=(5, 2))

            net = Network(n, cfg["n_communities"], cfg["within_bias"])

            placement = {}
            if cfg["user_communities"] is not None:
                placement["rare"] = cfg["user_communities"]

            world = World(
                constructions=[cxn],
                n_agents=n,
                rho=args.rho,
                network=net,
                user_placement=placement,
                random_seed=seed,
            )
            world.run_and_record(steps, interact, record_every)

            traj = [s["rare_mean"] for s in world.history]
            ptraj = [s["rare_prod"] for s in world.history]
            trajectories.append(traj)
            prod_trajectories.append(ptraj)

        # compute summary stats across runs
        n_points = min(len(t) for t in trajectories)
        mean_traj = []
        lo_traj = []
        hi_traj = []
        mean_prod = []

        for i in range(n_points):
            vals = [t[i] for t in trajectories]
            pvals = [t[i] for t in prod_trajectories]
            m = sum(vals) / len(vals)
            mean_traj.append(m)
            sorted_v = sorted(vals)
            lo_traj.append(sorted_v[max(0, len(sorted_v) // 10)])
            hi_traj.append(sorted_v[min(len(sorted_v) - 1,
                                        9 * len(sorted_v) // 10)])
            mean_prod.append(sum(pvals) / len(pvals))

        all_results[cond_name] = {
            "mean": mean_traj,
            "lo": lo_traj,
            "hi": hi_traj,
            "prod": mean_prod,
            "n_points": n_points,
        }

        final_m = mean_traj[-1]
        final_p = mean_prod[-1]
        print(f"    mean final confidence: {final_m:.3f}")
        print(f"    mean final producers:  {final_p:.0%}")

        # show per-community breakdown for last run
        if cfg["n_communities"] > 1:
            comm_stats = world.community_stats("rare")
            for cid in sorted(comm_stats.keys()):
                cm, cp = comm_stats[cid]
                marker = " <-- users" if (cfg["user_communities"]
                                          and cid in cfg["user_communities"]) else ""
                print(f"      community {cid}: conf={cm:.3f} "
                      f"prod={cp:.0%}{marker}")

    if HAS_MPL and not args.no_viz:
        plot_clustering(all_results, record_every, steps)

    return all_results


def experiment_rho_sensitivity(args):
    """
    THE AUDIT: how much does rho (the key free parameter) change everything?

    Three sub-experiments at each rho value:
      A. Critical mass grid: rho × initial_users → final producer fraction
      B. Clustering comparison: random_mixing vs clustered_1
      C. Bimodality check: rare_licensed distribution shape

    Pre-registered predictions (written before running):
      - Niche-speed relationship should hold at ALL rho values (it's mathematical)
      - Critical mass tipping point should EXIST at all rho, but MOVE with rho
      - Analytic lower bound on critical mass: f > rho / (1 + rho)
      - Dialectal pockets should form at all rho with community structure
      - Bimodality: uncertain. May be rho-dependent.
    """
    n = args.agents
    steps = args.steps
    interact = args.interactions or n
    n_runs = args.mc_runs
    niche = 0.10  # moderate niche for critical mass

    rho_values = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8]
    fractions = [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]

    print("=" * 60)
    print("RHO SENSITIVITY ANALYSIS")
    print(f"  {n} agents, {steps} steps, {interact} interactions/step")
    print(f"  {n_runs} MC runs per cell")
    print(f"  rho values: {rho_values}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # A. Critical mass grid: rho × initial_users
    # ------------------------------------------------------------------
    print("\n--- A. Critical mass × rho grid ---")
    print(f"  niche_width = {niche}")

    # grid[rho][frac] = (mean_conf, mean_prod) averaged over MC runs
    grid = {}
    analytic_bounds = {}

    for rho in rho_values:
        grid[rho] = {}
        analytic_bounds[rho] = rho / (1 + rho)
        print(f"\n  rho = {rho}  (analytic lower bound: f > {rho/(1+rho):.3f})")

        for frac in fractions:
            confs = []
            prods = []
            for run in range(n_runs):
                cxns = [Construction("target", niche_width=niche,
                                     initial_users=frac,
                                     initial_confidence=(6, 2))]
                world = World(constructions=cxns, n_agents=n, rho=rho,
                              random_seed=args.seed + run)
                world.run(n_steps=steps, interactions=interact)
                world.record_snapshot()
                confs.append(world.history[-1]["target_mean"])
                prods.append(world.history[-1]["target_prod"])

            mean_c = sum(confs) / len(confs)
            mean_p = sum(prods) / len(prods)
            grid[rho][frac] = (mean_c, mean_p)
            survived = "ALIVE" if mean_p > 0.5 else "dead"
            print(f"    f={frac:>5.0%}  conf={mean_c:.3f}  "
                  f"prod={mean_p:.0%}  [{survived}]")

    # ------------------------------------------------------------------
    # B. Clustering comparison at each rho
    # ------------------------------------------------------------------
    print("\n--- B. Clustering vs random mixing at each rho ---")
    print(f"  10% initial users, niche=0.04, 10 communities, beta=0.80")

    cluster_niche = 0.04
    cluster_results = {}

    for rho in rho_values:
        cluster_results[rho] = {}

        for cond_name, cond_cfg in [
            ("random", {"nc": 1, "wb": 0.0, "uc": None}),
            ("clustered", {"nc": 10, "wb": 0.80, "uc": [0]}),
        ]:
            prods_all = []
            for run in range(n_runs):
                cxn = Construction("rare", niche_width=cluster_niche,
                                   initial_users=0.10,
                                   initial_confidence=(5, 2))
                net = Network(n, cond_cfg["nc"], cond_cfg["wb"])
                placement = {"rare": cond_cfg["uc"]} if cond_cfg["uc"] else {}

                world = World(constructions=[cxn], n_agents=n, rho=rho,
                              network=net, user_placement=placement,
                              random_seed=args.seed + run)
                world.run(n_steps=steps, interactions=interact)
                world.record_snapshot()
                prods_all.append(world.history[-1]["rare_prod"])

            mean_p = sum(prods_all) / len(prods_all)
            cluster_results[rho][cond_name] = mean_p

        print(f"  rho={rho}:  random={cluster_results[rho]['random']:.0%}  "
              f"clustered={cluster_results[rho]['clustered']:.0%}  "
              f"diff={cluster_results[rho]['random'] - cluster_results[rho]['clustered']:+.0%}")

    # ------------------------------------------------------------------
    # C. Bimodality check at each rho
    # ------------------------------------------------------------------
    print("\n--- C. Bimodality check (rare_licensed, niche=0.04, 10% users) ---")

    bimod_worlds = {}
    for rho in rho_values:
        cxns = [Construction("rare", niche_width=0.04, initial_users=0.10,
                             initial_confidence=(5, 2))]
        world = World(constructions=cxns, n_agents=n, rho=rho,
                       random_seed=args.seed)
        world.run(n_steps=steps, interactions=interact)

        confs = sorted([a.beliefs["rare"].confidence for a in world.agents])
        # quick bimodality check: compare density in bottom/top thirds vs middle
        n_lo = sum(1 for c in confs if c < 0.35)
        n_mid = sum(1 for c in confs if 0.35 <= c <= 0.65)
        n_hi = sum(1 for c in confs if c > 0.65)
        bimodal = (n_lo > n * 0.15 and n_hi > n * 0.15
                   and n_mid < max(n_lo, n_hi))
        median_c = confs[len(confs) // 2]

        bimod_worlds[rho] = world
        print(f"  rho={rho}:  median={median_c:.3f}  "
              f"lo={n_lo/n:.0%}  mid={n_mid/n:.0%}  hi={n_hi/n:.0%}  "
              f"{'BIMODAL' if bimodal else 'unimodal'}")

    if HAS_MPL and not args.no_viz:
        plot_rho_heatmap(rho_values, fractions, grid, analytic_bounds)
        plot_rho_clustering(rho_values, cluster_results)
        plot_rho_distributions(rho_values, bimod_worlds)

    return grid, cluster_results, bimod_worlds


# ============================================================================
# VISUALIZATION
# ============================================================================

COLORS = {
    "core": "#2ecc71",
    "established": "#3498db",
    "rare_licensed": "#f39c12",
    "stable_gap": "#e74c3c",
    "community_novel": "#9b59b6",
    "emerging": "#1abc9c",
    "target": "#e67e22",
}

COND_COLORS = {
    "random_mixing": "#95a5a6",
    "scattered": "#e74c3c",
    "clustered_2": "#f39c12",
    "clustered_1": "#2ecc71",
}


def plot_trajectories(world: World, filename: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    times = [s["time"] for s in world.history]

    for name in world.constructions:
        color = COLORS.get(name, "#333333")
        means = [s[f"{name}_mean"] for s in world.history]
        stds = [s[f"{name}_std"] for s in world.history]
        prods = [s[f"{name}_prod"] for s in world.history]

        ax1.plot(times, means, color=color, linewidth=2, label=name)
        ax1.fill_between(times,
                         [m - s for m, s in zip(means, stds)],
                         [m + s for m, s in zip(means, stds)],
                         color=color, alpha=0.12)
        ax2.plot(times, prods, color=color, linewidth=2, label=name)

    ax1.set_ylabel("Mean confidence  c = a/(a+b)", fontsize=11)
    ax1.set_title("Population Grammaticality Dynamics\n"
                  f"({world.n_agents} agents, endogenous preemption)",
                  fontsize=13)
    ax1.legend(loc="center right", fontsize=9)
    ax1.set_ylim(-0.02, 1.02)
    ax1.axhline(0.5, color="gray", ls=":", alpha=0.4)
    ax1.grid(True, alpha=0.2)

    ax2.set_xlabel("Time step", fontsize=11)
    ax2.set_ylabel("Fraction of producers  (conf > 0.5)", fontsize=11)
    ax2.set_ylim(-0.02, 1.02)
    ax2.axhline(0.5, color="gray", ls=":", alpha=0.4)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"\nPlot saved: {filename}")
    plt.close()


def plot_distributions(world: World, filename: str):
    names = list(world.constructions.keys())
    ncols = min(3, len(names))
    nrows = math.ceil(len(names) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if nrows * ncols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, name in enumerate(names):
        ax = axes[idx]
        confs = [a.beliefs[name].confidence for a in world.agents]
        color = COLORS.get(name, "#333333")
        ax.hist(confs, bins=40, color=color, alpha=0.7, edgecolor="white")
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Confidence")
        ax.set_xlim(0, 1)
        ax.axvline(0.5, color="gray", ls=":", alpha=0.5)

    for idx in range(len(names), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"Agent confidence distributions at t={world.time_step} "
                 f"({world.n_agents} agents)", fontsize=13)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Plot saved: {filename}")
    plt.close()


def plot_critical_mass(fractions, results):
    fig, ax = plt.subplots(figsize=(9, 5))
    means = [results[f][0] for f in fractions]
    prods = [results[f][1] for f in fractions]
    pcts = [f * 100 for f in fractions]

    ax.plot(pcts, means, "o-", color="#e67e22", linewidth=2,
            markersize=8, label="Final confidence")
    ax.plot(pcts, prods, "s--", color="#3498db", linewidth=2,
            markersize=8, label="Final producer fraction")

    ax.set_xlabel("Initial user fraction (%)", fontsize=11)
    ax.set_ylabel("Value", fontsize=11)
    ax.set_title("Critical Mass: How Many Seed Users Does a Construction Need?",
                 fontsize=13)
    ax.legend()
    ax.set_ylim(-0.02, 1.02)
    ax.axhline(0.5, color="gray", ls=":", alpha=0.4)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig("critical_mass_sweep.png", dpi=150)
    print(f"\nPlot saved: critical_mass_sweep.png")
    plt.close()


def plot_niche_sweep(all_histories):
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.RdYlGn_r

    widths = sorted(all_histories.keys())
    for i, w in enumerate(widths):
        color = cmap(i / max(1, len(widths) - 1))
        traj = all_histories[w]
        ax.plot(range(1, len(traj) + 1), traj, color=color,
                linewidth=2, label=f"niche={w:.3f}")

    ax.set_xlabel("Time step (recording interval)", fontsize=11)
    ax.set_ylabel("Mean confidence", fontsize=11)
    ax.set_title("Niche Width and Death Speed\n"
                 "Wide niche = fast preemption; narrow niche = slow decay",
                 fontsize=13)
    ax.legend(fontsize=9)
    ax.set_ylim(-0.02, 1.02)
    ax.axhline(0.5, color="gray", ls=":", alpha=0.4)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig("niche_sweep.png", dpi=150)
    print(f"\nPlot saved: niche_sweep.png")
    plt.close()


def plot_clustering(all_results, record_every, total_steps):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    for cond_name, data in all_results.items():
        color = COND_COLORS.get(cond_name, "#333333")
        n_pts = data["n_points"]
        times = [record_every * (i + 1) for i in range(n_pts)]

        ax1.plot(times, data["mean"], color=color, linewidth=2.5,
                 label=cond_name)
        ax1.fill_between(times, data["lo"], data["hi"],
                         color=color, alpha=0.15)

        ax2.plot(times, data["prod"], color=color, linewidth=2.5,
                 label=cond_name)

    ax1.set_ylabel("Mean confidence", fontsize=11)
    ax1.set_title("Does Network Clustering Save Rare Forms?\n"
                  "Same 10% initial users, different spatial arrangements",
                  fontsize=13)
    ax1.legend(fontsize=10)
    ax1.set_ylim(-0.02, 1.02)
    ax1.axhline(0.5, color="gray", ls=":", alpha=0.4)
    ax1.grid(True, alpha=0.2)

    ax2.set_xlabel("Time step", fontsize=11)
    ax2.set_ylabel("Fraction of producers  (conf > 0.5)", fontsize=11)
    ax2.set_ylim(-0.02, 1.02)
    ax2.axhline(0.5, color="gray", ls=":", alpha=0.4)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig("clustering_experiment.png", dpi=150)
    print(f"\nPlot saved: clustering_experiment.png")
    plt.close()


def plot_rho_heatmap(rho_values, fractions, grid, analytic_bounds):
    """2D heatmap: rho × initial_users → final producer fraction."""
    fig, ax = plt.subplots(figsize=(11, 6))

    # build matrix
    mat = []
    for rho in rho_values:
        row = [grid[rho][f][1] for f in fractions]  # producer fraction
        mat.append(row)
    mat = np.array(mat)

    im = ax.imshow(mat, aspect='auto', origin='lower', cmap='RdYlGn',
                   vmin=0, vmax=1,
                   extent=[-0.5, len(fractions) - 0.5,
                           -0.5, len(rho_values) - 0.5])

    # annotate cells
    for i, rho in enumerate(rho_values):
        for j, frac in enumerate(fractions):
            val = mat[i, j]
            color = "white" if val < 0.3 or val > 0.7 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold")

    ax.set_xticks(range(len(fractions)))
    ax.set_xticklabels([f"{f:.0%}" for f in fractions])
    ax.set_yticks(range(len(rho_values)))
    ax.set_yticklabels([f"{r}" for r in rho_values])
    ax.set_xlabel("Initial user fraction", fontsize=11)
    ax.set_ylabel("rho (preemption weight)", fontsize=11)

    # overlay analytic lower bound
    for i, rho in enumerate(rho_values):
        bound = analytic_bounds[rho]
        # find x position of bound in fractions axis
        for j, f in enumerate(fractions):
            if f >= bound:
                ax.plot(j - 0.5, i, '|', color='black', markersize=20,
                        markeredgewidth=2)
                break

    ax.set_title("Critical Mass × Preemption Weight\n"
                 "Green = form survives. Red = form dies. "
                 "Black marks = analytic lower bound f > rho/(1+rho)",
                 fontsize=12)

    fig.colorbar(im, ax=ax, label="Final producer fraction", shrink=0.8)
    plt.tight_layout()
    plt.savefig("rho_sensitivity_heatmap.png", dpi=150)
    print(f"\nPlot saved: rho_sensitivity_heatmap.png")
    plt.close()


def plot_rho_clustering(rho_values, cluster_results):
    """Bar chart: random vs clustered at each rho."""
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(rho_values))
    width = 0.35
    random_vals = [cluster_results[r]["random"] for r in rho_values]
    clustered_vals = [cluster_results[r]["clustered"] for r in rho_values]

    bars1 = ax.bar(x - width/2, random_vals, width, label="Random mixing",
                   color="#95a5a6", edgecolor="white")
    bars2 = ax.bar(x + width/2, clustered_vals, width, label="Clustered",
                   color="#2ecc71", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in rho_values])
    ax.set_xlabel("rho (preemption weight)", fontsize=11)
    ax.set_ylabel("Final producer fraction (global)", fontsize=11)
    ax.set_title("Random Mixing vs Clustering Across Rho Values\n"
                 "10% initial users, niche=0.04, 10 communities",
                 fontsize=12)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", ls=":", alpha=0.4)
    ax.grid(True, alpha=0.2, axis='y')

    # annotate differences
    for i, rho in enumerate(rho_values):
        diff = random_vals[i] - clustered_vals[i]
        y = max(random_vals[i], clustered_vals[i]) + 0.03
        ax.text(i, y, f"{diff:+.0%}", ha="center", fontsize=8,
                color="#555")

    plt.tight_layout()
    plt.savefig("rho_clustering_comparison.png", dpi=150)
    print(f"Plot saved: rho_clustering_comparison.png")
    plt.close()


def plot_rho_distributions(rho_values, bimod_worlds):
    """Histograms of agent confidence at each rho (bimodality check)."""
    n_rho = len(rho_values)
    ncols = 3
    nrows = math.ceil(n_rho / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    cmap = plt.cm.coolwarm
    for idx, rho in enumerate(rho_values):
        ax = axes[idx]
        world = bimod_worlds[rho]
        confs = [a.beliefs["rare"].confidence for a in world.agents]
        color = cmap(idx / max(1, n_rho - 1))
        ax.hist(confs, bins=40, color=color, alpha=0.7, edgecolor="white")
        ax.set_title(f"rho = {rho}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Confidence")
        ax.set_xlim(0, 1)
        ax.axvline(0.5, color="gray", ls=":", alpha=0.5)

    for idx in range(n_rho, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Rare-Licensed Confidence Distributions Across Rho\n"
                 "(10% initial users, niche=0.04, 1000 agents)",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig("rho_bimodality.png", dpi=150)
    print(f"Plot saved: rho_bimodality.png")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Grammaticality Population Dynamics ABM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Experiments:
  baseline  Scaled-up baseline with default constructions
  sweep     Critical-mass sweep: vary initial user fraction
  niche     Niche-width sweep: how fast do unused forms die?
  cluster   Network clustering: does community structure save rare forms?
  rho       RHO SENSITIVITY: the audit (heatmap, clustering, bimodality)
  all       Run everything
        """)
    parser.add_argument("--exp", default="baseline")
    parser.add_argument("--agents", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--interactions", type=int, default=0,
                        help="Interactions per step (default: n_agents)")
    parser.add_argument("--rho", type=float, default=0.3,
                        help="Preemption weight")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mc-runs", type=int, default=20,
                        help="Monte Carlo runs for clustering experiment")
    parser.add_argument("--no-viz", action="store_true")

    args = parser.parse_args()

    t_total = time.time()

    if args.exp in ("baseline", "all"):
        experiment_baseline(args)
    if args.exp in ("sweep", "all"):
        experiment_critical_mass(args)
    if args.exp in ("niche", "all"):
        experiment_niche_sweep(args)
    if args.exp in ("cluster", "all"):
        experiment_clustering(args)
    if args.exp in ("rho", "all"):
        experiment_rho_sensitivity(args)
    if args.exp not in ("baseline", "sweep", "niche", "cluster", "rho", "all"):
        print(f"Unknown experiment: {args.exp}")
        parser.print_help()

    print(f"\nTotal wall time: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
