#!/usr/bin/env python3
"""
Window Sweep Calibration
=========================

PURPOSE: The original calibration (fake_data_calibration.py) showed a ceiling
effect: the selection window [0.20, 0.50] cannot distinguish rho=0.5 from
rho=0.7 from rho=1.0. Can a wider window fix this?

PROTOCOL:
---------

Design: Same ABM setup as fake_data_calibration.py:
  True rho: {0.2, 0.3, 0.5, 0.7, 1.0}
  258 features, U(0.01, 0.99) initial fractions
  200 agents, 200 steps, niche=0.10
  20 MC replicates per rho

Key innovation: Run ABMs ONCE, then apply multiple selection windows to
the same cached data. This eliminates simulation noise as a confound
when comparing windows.

Windows to test:
  [0.20, 0.50]  (original)
  [0.15, 0.55]  (modest expansion)
  [0.10, 0.60]  (wider)
  [0.10, 0.70]  (much wider)
  [0.05, 0.80]  (very wide, borderline "use everything declining")

PRE-REGISTERED PREDICTIONS (written before running):
----------------------------------------------------

1. WIDER WINDOWS REDUCE THE CEILING. The ceiling comes from excluding
   features near the tipping point when the tipping point > 0.50.
   [0.10, 0.70] should capture features with young_rate up to 0.70,
   allowing rho estimates up to 0.70/(1-0.70) = 2.33. This should
   improve recovery at rho=0.7 and rho=1.0.

2. WIDER WINDOWS INCREASE NOISE. More non-tipping features pass the
   filter (declining features that are well below or above the tipping
   point). IQRs should widen with window width.

3. BIAS-VARIANCE TRADEOFF. There exists some intermediate window that
   minimizes total error (bias^2 + variance). The optimal window is
   unknown and probably depends on the true rho.

4. NO WINDOW FULLY FIXES rho=1.0 RECOVERY. At rho=1.0, the tipping
   point is 0.50. Features near the tipping point are being pushed
   rapidly toward 0 or 1 by the ABM dynamics. By the "young" snapshot,
   most will have resolved. The few caught mid-transition give
   rho_hat ~ 1.0, but there won't be many. Recovery will improve
   but remain noisy.

5. THE ORIGINAL WINDOW IS NOT OPTIMAL. It was chosen to match the
   SCOSYA analysis (features with 20-50% young acceptance). A wider
   window is likely better for estimation, even if it includes some
   irrelevant features.

WHAT WOULD CHANGE OUR CONCLUSIONS:
-----------------------------------

If a wider window makes rho=0.5, 0.7, and 1.0 distinguishable, then
the ceiling was an artifact of the window choice. We should re-estimate
rho from SCOSYA using the better window and report both the original
and corrected estimates.

If NO window fixes the ceiling, then the limitation is fundamental:
the estimation method (select declining features, compute f/(1-f))
cannot recover high rho regardless of the window. The ABM dynamics
erase the tipping-point signal too quickly.

Usage:
    python models/window_sweep_calibration.py
    python models/window_sweep_calibration.py --mc-runs 10
"""

import random
import time
import statistics
import argparse
import json
from typing import List, Tuple, Optional, Dict

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grammaticality_abm import Construction, World

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

TRUE_RHO_VALUES = [0.2, 0.3, 0.5, 0.7, 1.0]
N_FEATURES = 258
N_AGENTS = 200
N_STEPS = 200
OLD_SNAPSHOT = 100
YOUNG_SNAPSHOT = 200
INTERACTIONS = 200
NICHE_WIDTH = 0.10
CONFIDENCE_THRESHOLD = 0.5
DECLINE_THRESHOLD = -0.01

WINDOWS = [
    (0.20, 0.50),  # original
    (0.15, 0.55),  # modest expansion
    (0.10, 0.60),  # wider
    (0.10, 0.70),  # much wider
    (0.05, 0.80),  # very wide
]


# ============================================================================
# SIMULATION (same as fake_data_calibration.py)
# ============================================================================

def generate_feature_population(n_features: int, seed: int) -> List[float]:
    rng = random.Random(seed)
    return [rng.uniform(0.01, 0.99) for _ in range(n_features)]


def run_single_feature(initial_users: float, rho: float,
                       seed: int) -> Tuple[float, float]:
    cxn = Construction(
        name="feature", niche_width=NICHE_WIDTH,
        initial_users=initial_users, initial_confidence=(6, 2))
    world = World(constructions=[cxn], n_agents=N_AGENTS,
                  rho=rho, random_seed=seed)

    world.run(n_steps=OLD_SNAPSHOT, interactions=INTERACTIONS)
    old_rate = sum(1 for a in world.agents
                   if a.beliefs["feature"].confidence > CONFIDENCE_THRESHOLD) / N_AGENTS

    world.run(n_steps=YOUNG_SNAPSHOT - OLD_SNAPSHOT, interactions=INTERACTIONS)
    young_rate = sum(1 for a in world.agents
                     if a.beliefs["feature"].confidence > CONFIDENCE_THRESHOLD) / N_AGENTS

    return old_rate, young_rate


def run_all_abms(n_replicates: int) -> Dict[float, List[List[Tuple[float, float, float]]]]:
    """Run all ABMs once and cache the features_data."""
    cached = {}
    for rho_idx, true_rho in enumerate(TRUE_RHO_VALUES):
        base_seed = (rho_idx + 1) * 100000
        rho_t0 = time.time()
        print(f"\n  Running ABMs for rho={true_rho} "
              f"(critical mass={true_rho/(1+true_rho):.3f})...")

        reps = []
        for rep in range(n_replicates):
            seed_offset = base_seed + rep * 10000
            initial_fracs = generate_feature_population(N_FEATURES, seed_offset)
            features_data = []
            for i, init_f in enumerate(initial_fracs):
                old_rate, young_rate = run_single_feature(
                    init_f, true_rho, seed_offset + i + 1)
                features_data.append((init_f, old_rate, young_rate))
            reps.append(features_data)
            if (rep + 1) % 5 == 0:
                print(f"    {rep+1}/{n_replicates}")

        cached[true_rho] = reps
        print(f"    [{time.time() - rho_t0:.1f}s]")

    return cached


# ============================================================================
# ESTIMATION (parameterized by window)
# ============================================================================

def estimate_rho_with_window(features_data, window):
    """Apply estimation method with a given selection window."""
    lo, hi = window
    qualifying = []

    for init_f, old_rate, young_rate in features_data:
        delta = young_rate - old_rate
        if delta < DECLINE_THRESHOLD and lo <= young_rate <= hi:
            rho_est = young_rate / (1 - young_rate)
            qualifying.append(rho_est)

    if not qualifying:
        return None, 0

    return statistics.median(qualifying), len(qualifying)


def evaluate_window(cached_data, window):
    """Evaluate one window across all rho values and replicates."""
    results = {}
    for true_rho in TRUE_RHO_VALUES:
        estimates = []
        n_quals = []
        for features_data in cached_data[true_rho]:
            rho_hat, n_qual = estimate_rho_with_window(features_data, window)
            estimates.append(rho_hat)
            n_quals.append(n_qual)

        valid = [e for e in estimates if e is not None]
        n_nan = sum(1 for e in estimates if e is None)

        if valid:
            med = statistics.median(valid)
            q25 = sorted(valid)[max(0, len(valid) // 4)]
            q75 = sorted(valid)[min(len(valid) - 1, 3 * len(valid) // 4)]
            results[true_rho] = {
                "median": med,
                "bias": med - true_rho,
                "q25": q25, "q75": q75,
                "iqr_width": q75 - q25,
                "med_nqual": statistics.median(n_quals),
                "n_nan": n_nan,
                "estimates": valid,
            }
        else:
            results[true_rho] = {
                "median": None, "bias": None,
                "q25": None, "q75": None, "iqr_width": None,
                "med_nqual": statistics.median(n_quals),
                "n_nan": n_nan, "estimates": [],
            }

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Window sweep calibration")
    parser.add_argument("--mc-runs", type=int, default=20)
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("WINDOW SWEEP CALIBRATION")
    print("=" * 70)
    print(f"\n  Windows: {WINDOWS}")
    print(f"  True rho: {TRUE_RHO_VALUES}")
    print(f"  MC replicates: {args.mc_runs}")
    print(f"  Total ABM runs: {len(TRUE_RHO_VALUES) * N_FEATURES * args.mc_runs}")

    # Phase 1: Run all ABMs (the expensive part)
    print(f"\n{'='*70}")
    print("PHASE 1: Running ABMs (one-time cost)")
    print(f"{'='*70}")
    t0 = time.time()
    cached = run_all_abms(args.mc_runs)
    abm_time = time.time() - t0
    print(f"\n  ABM phase complete: {abm_time:.1f}s")

    # Phase 2: Apply each window (cheap)
    print(f"\n{'='*70}")
    print("PHASE 2: Evaluating windows")
    print(f"{'='*70}")

    all_window_results = {}
    for window in WINDOWS:
        all_window_results[window] = evaluate_window(cached, window)

    # ===================================================================
    # RESULTS: comparison table
    # ===================================================================
    print(f"\n{'='*70}")
    print("RESULTS: RECOVERY BY WINDOW AND TRUE RHO")
    print(f"{'='*70}")

    for true_rho in TRUE_RHO_VALUES:
        print(f"\n  --- True rho = {true_rho} (critical mass = "
              f"{true_rho/(1+true_rho):.3f}) ---")
        print(f"  {'Window':>16} {'Median':>8} {'Bias':>8} "
              f"{'IQR':>16} {'IQR width':>10} {'n_qual':>8} {'NaN':>5}")
        print(f"  {'-'*73}")

        for window in WINDOWS:
            r = all_window_results[window][true_rho]
            wstr = f"[{window[0]:.2f}, {window[1]:.2f}]"
            if r["median"] is not None:
                print(f"  {wstr:>16} {r['median']:>8.3f} {r['bias']:>+8.3f} "
                      f"[{r['q25']:.3f}, {r['q75']:.3f}] "
                      f"{r['iqr_width']:>10.3f} {r['med_nqual']:>8.0f} "
                      f"{r['n_nan']:>5}")
            else:
                print(f"  {wstr:>16} {'NaN':>8} {'':>8} "
                      f"{'':>16} {'':>10} {r['med_nqual']:>8.0f} "
                      f"{r['n_nan']:>5}")

    # ===================================================================
    # DISTINGUISHABILITY: can each window separate adjacent rho values?
    # ===================================================================
    print(f"\n{'='*70}")
    print("DISTINGUISHABILITY BY WINDOW")
    print(f"{'='*70}")

    pairs = [(0.2, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)]
    print(f"\n  {'Window':>16}", end="")
    for r1, r2 in pairs:
        print(f"  {r1} vs {r2}", end="")
    print()
    print(f"  {'-'*70}")

    for window in WINDOWS:
        wstr = f"[{window[0]:.2f}, {window[1]:.2f}]"
        print(f"  {wstr:>16}", end="")
        wr = all_window_results[window]
        for r1, r2 in pairs:
            s1, s2 = wr[r1], wr[r2]
            if (s1["q25"] is not None and s2["q25"] is not None):
                overlap = s1["q75"] > s2["q25"]
                print(f"  {'OVERLAP' if overlap else '  SEP  ':>9}", end="")
            else:
                print(f"  {'NaN':>9}", end="")
        print()

    # ===================================================================
    # SUMMARY SCORES: which window is "best"?
    # ===================================================================
    print(f"\n{'='*70}")
    print("SUMMARY: MEAN ABSOLUTE BIAS AND MEAN IQR WIDTH")
    print(f"{'='*70}")

    print(f"\n  {'Window':>16} {'Mean |bias|':>12} {'Mean IQR':>10} "
          f"{'Separated':>10} {'Total NaN':>10}")
    print(f"  {'-'*58}")

    for window in WINDOWS:
        wr = all_window_results[window]
        wstr = f"[{window[0]:.2f}, {window[1]:.2f}]"

        biases = [abs(wr[r]["bias"]) for r in TRUE_RHO_VALUES
                  if wr[r]["bias"] is not None]
        iqrs = [wr[r]["iqr_width"] for r in TRUE_RHO_VALUES
                if wr[r]["iqr_width"] is not None]
        total_nan = sum(wr[r]["n_nan"] for r in TRUE_RHO_VALUES)

        # Count separated pairs
        n_sep = 0
        for r1, r2 in pairs:
            s1, s2 = wr[r1], wr[r2]
            if (s1["q25"] is not None and s2["q25"] is not None
                    and s1["q75"] <= s2["q25"]):
                n_sep += 1

        mean_bias = statistics.mean(biases) if biases else float('nan')
        mean_iqr = statistics.mean(iqrs) if iqrs else float('nan')

        print(f"  {wstr:>16} {mean_bias:>12.3f} {mean_iqr:>10.3f} "
              f"{n_sep:>10}/4 {total_nan:>10}")

    # ===================================================================
    # VERDICT
    # ===================================================================
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")

    # Check predictions
    orig = all_window_results[(0.20, 0.50)]
    wide = all_window_results[(0.10, 0.70)]

    # P1: wider reduces ceiling
    if (orig[0.7]["bias"] is not None and wide[0.7]["bias"] is not None):
        p1 = abs(wide[0.7]["bias"]) < abs(orig[0.7]["bias"])
        print(f"\n  P1 (wider reduces ceiling at rho=0.7): "
              f"orig bias={orig[0.7]['bias']:+.3f}, "
              f"wide bias={wide[0.7]['bias']:+.3f} -> "
              f"{'CONFIRMED' if p1 else 'REJECTED'}")

    # P2: wider increases noise
    if (orig[0.5]["iqr_width"] is not None and wide[0.5]["iqr_width"] is not None):
        p2 = wide[0.5]["iqr_width"] > orig[0.5]["iqr_width"]
        print(f"  P2 (wider increases noise at rho=0.5): "
              f"orig IQR={orig[0.5]['iqr_width']:.3f}, "
              f"wide IQR={wide[0.5]['iqr_width']:.3f} -> "
              f"{'CONFIRMED' if p2 else 'REJECTED'}")

    # P4: no window fully fixes rho=1.0
    best_10_bias = min(
        abs(all_window_results[w][1.0]["bias"])
        for w in WINDOWS
        if all_window_results[w][1.0]["bias"] is not None
    )
    p4 = best_10_bias > 0.15
    print(f"  P4 (no window fully fixes rho=1.0): "
          f"best |bias|={best_10_bias:.3f} -> "
          f"{'CONFIRMED' if p4 else 'REJECTED'}")

    # P5: original is not optimal
    orig_mean_bias = statistics.mean(
        [abs(orig[r]["bias"]) for r in TRUE_RHO_VALUES
         if orig[r]["bias"] is not None])
    best_window = None
    best_mean_bias = orig_mean_bias
    for window in WINDOWS:
        wr = all_window_results[window]
        mb = statistics.mean(
            [abs(wr[r]["bias"]) for r in TRUE_RHO_VALUES
             if wr[r]["bias"] is not None])
        if mb < best_mean_bias:
            best_mean_bias = mb
            best_window = window
    p5 = best_window is not None
    if best_window:
        print(f"  P5 (original not optimal): "
              f"best window={best_window} "
              f"(mean |bias|={best_mean_bias:.3f} vs "
              f"orig={orig_mean_bias:.3f}) -> CONFIRMED")
    else:
        print(f"  P5 (original not optimal): "
              f"original IS the best -> REJECTED")

    # The big question
    print(f"\n  KEY QUESTION: Does any window separate rho=0.5 from rho=0.7?")
    for window in WINDOWS:
        wr = all_window_results[window]
        s5, s7 = wr[0.5], wr[0.7]
        if s5["q25"] is not None and s7["q25"] is not None:
            overlap = s5["q75"] > s7["q25"]
            wstr = f"[{window[0]:.2f}, {window[1]:.2f}]"
            print(f"    {wstr}: [{s5['q25']:.3f}, {s5['q75']:.3f}] vs "
                  f"[{s7['q25']:.3f}, {s7['q75']:.3f}] -> "
                  f"{'OVERLAP' if overlap else 'SEPARATED'}")

    print(f"\n  Total wall time: {time.time() - t0:.1f}s")

    # ===================================================================
    # VISUALIZATION
    # ===================================================================
    if HAS_MPL and not args.no_viz:
        plot_window_comparison(all_window_results)


def plot_window_comparison(all_window_results):
    """Recovery curves for each window."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = ["#e74c3c", "#e67e22", "#2ecc71", "#3498db", "#9b59b6"]

    # Panel 1: True vs estimated rho
    ax = axes[0]
    ax.plot([0, 1.2], [0, 1.2], "k--", alpha=0.3, label="Perfect")

    for idx, window in enumerate(WINDOWS):
        wr = all_window_results[window]
        rhos = [r for r in TRUE_RHO_VALUES if wr[r]["median"] is not None]
        meds = [wr[r]["median"] for r in rhos]
        q25s = [wr[r]["q25"] for r in rhos]
        q75s = [wr[r]["q75"] for r in rhos]
        wstr = f"[{window[0]:.2f}, {window[1]:.2f}]"

        ax.errorbar(
            [r + (idx - 2) * 0.02 for r in rhos], meds,
            yerr=[[m - q for m, q in zip(meds, q25s)],
                  [q - m for m, q in zip(meds, q75s)]],
            fmt="o-", color=colors[idx], linewidth=1.5, markersize=6,
            capsize=4, label=wstr, alpha=0.8)

    ax.set_xlabel("True rho", fontsize=11)
    ax.set_ylabel("Estimated rho", fontsize=11)
    ax.set_title("Recovery by Selection Window", fontsize=12)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(0.1, 1.1)
    ax.set_ylim(0, 1.5)
    ax.grid(True, alpha=0.2)

    # Panel 2: Bias by window and rho
    ax = axes[1]
    x = range(len(TRUE_RHO_VALUES))
    bar_width = 0.15

    for idx, window in enumerate(WINDOWS):
        wr = all_window_results[window]
        biases = [wr[r]["bias"] if wr[r]["bias"] is not None else 0
                  for r in TRUE_RHO_VALUES]
        wstr = f"[{window[0]:.2f}, {window[1]:.2f}]"
        ax.bar([xi + (idx - 2) * bar_width for xi in x], biases,
               bar_width, color=colors[idx], alpha=0.7, label=wstr)

    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in TRUE_RHO_VALUES])
    ax.set_xlabel("True rho", fontsize=11)
    ax.set_ylabel("Bias (median estimate - true)", fontsize=11)
    ax.set_title("Estimation Bias by Window", fontsize=12)
    ax.legend(fontsize=8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("Window Sweep: Can a Wider Selection Window Fix the Ceiling?",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("window_sweep_calibration.png", dpi=150)
    print(f"\nPlot saved: window_sweep_calibration.png")
    plt.close()


if __name__ == "__main__":
    main()
