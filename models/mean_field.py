#!/usr/bin/env python3
"""
Mean-field analysis of the grammaticality population dynamics ABM.

Derives the critical mass analytically and compares to simulation.

The question: given rho, what fraction of initial users f* is needed
for a construction to survive? Can we predict this without simulation?

Mean-field approximation: replace the stochastic population with a
deterministic system tracking the average confidence of producers
and non-producers, plus the producer fraction p.
"""

import math

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def analytic_lower_bound(rho: float) -> float:
    """
    Simplest bound: f > rho / (1 + rho).

    Derivation:
        For a marginal agent (confidence c = tau = 0.5):
        - Rate of positive evidence: w * p * c_bar_prod * (1/step)
        - Rate of negative evidence: w * (1 - p * c_bar_prod) * rho * (1/step)

        For confidence to increase at the margin:
            w * p * c_bar_prod * (1-c) > w * (1 - p * c_bar_prod) * rho * c

        At c = 0.5 this simplifies to:
            p * c_bar_prod > (1 - p * c_bar_prod) * rho
            p * c_bar_prod * (1 + rho) > rho
            p * c_bar_prod > rho / (1 + rho)

        If producers are fully confident (c_bar_prod ≈ 1):
            p > rho / (1 + rho)

    This is a lower bound because it assumes producers always produce.
    In reality, they produce with probability c_bar_prod < 1, so
    the actual threshold is higher.
    """
    return rho / (1 + rho)


def mean_field_critical_mass(rho: float, tau: float = 0.5,
                              c_prod: float = 0.75) -> float:
    """
    Tighter estimate: account for producers having finite confidence.

    At the tipping point, a marginal agent (c = tau) must have
    increasing confidence. Producers produce with probability c_prod
    (not 1.0). So the effective positive-evidence rate is p * c_prod.

    Condition: p * c_prod > rho / (1 + rho)
    Therefore: p > rho / (c_prod * (1 + rho))

    The remaining question is: what is c_prod at the tipping point?
    For the initial seed users with Beta(a0, b0) = Beta(6, 2),
    c_prod ≈ a0/(a0+b0) = 0.75. After some dynamics, producers
    who survive should be higher (positive feedback), but near the
    tipping point they haven't had much reinforcement yet.
    """
    return rho / (c_prod * (1 + rho))


def mean_field_trajectory(f: float, rho: float, w: float = 0.10,
                           tau: float = 0.5, c_init_user: float = 0.75,
                           c_init_nonuser: float = 0.5,
                           n_steps: int = 300) -> list:
    """
    Deterministic mean-field trajectory.

    Track two sub-populations:
        - Users (fraction f): start at c_init_user
        - Non-users (fraction 1-f): start at c_init_nonuser

    Each timestep, update both using the mean-field rates.
    Agents with c > tau produce; others don't.

    Returns list of (c_user, c_nonuser, p) per step.
    """
    c_u = c_init_user   # user confidence
    c_n = c_init_nonuser  # non-user confidence

    trajectory = []

    for t in range(n_steps):
        # producer fraction: users who produce + non-users who produce
        # produce iff c > tau, then with probability c
        p_u = c_u if c_u > tau else 0.0  # production prob per user
        p_n = c_n if c_n > tau else 0.0  # production prob per non-user

        # effective production rate (what a hearer encounters)
        prod_rate = f * p_u + (1 - f) * p_n

        # fraction of population that are "producers" (c > tau)
        frac_prod = 0.0
        if c_u > tau:
            frac_prod += f
        if c_n > tau:
            frac_prod += (1 - f)

        trajectory.append((c_u, c_n, frac_prod))

        # --- update confidences ---
        # For each sub-population, compute dc/dt from evidence rates.
        # Per interaction where agent is hearer:
        #   E[delta_a] = w * prod_rate
        #   E[delta_b] = w * (1 - prod_rate) * rho
        #
        # For Beta(a,b), dc/dt = (da*b - db*a) / (a+b)^2
        # In the mean-field, we track c directly using the
        # continuous approximation:
        #   dc/dt = (1-c) * rate_positive - c * rate_negative
        #         = (1-c) * w * prod_rate - c * w * (1-prod_rate) * rho
        #
        # (This follows from d/dt[a/(a+b)] with large a+b.)
        # Scale by 1/(a+b) to account for increasing belief strength.
        # Use a time-dependent (a+b) that grows as evidence accumulates.

        # Approximate (a+b) growth: starts at ~10 for users, ~2 for non-users
        # grows by w * 1 per step (each step adds some evidence)
        strength_u = 10 + t * w * 1.5  # rough
        strength_n = 2 + t * w * 1.5

        rate_pos = w * prod_rate
        rate_neg = w * (1 - prod_rate) * rho

        dc_u = ((1 - c_u) * rate_pos - c_u * rate_neg) / strength_u
        dc_n = ((1 - c_n) * rate_pos - c_n * rate_neg) / strength_n

        c_u = max(0, min(1, c_u + dc_u))
        c_n = max(0, min(1, c_n + dc_n))

    return trajectory


def compare_bounds_to_simulation():
    """
    Compare analytic predictions to the simulation results from the
    rho sensitivity experiment (hardcoded from the run).
    """
    # Simulation results: (rho, observed critical mass range)
    # "critical mass" = smallest f where producers > 50%
    sim_data = {
        0.05: (0.02, 1.00),  # 2% seed → 100% producers
        0.10: (0.02, 1.00),
        0.20: (0.05, 0.99),  # 2% dead, 5% alive
        0.30: (0.08, 0.54),  # 5% dead, 8% at 54% (borderline)
        0.50: (0.30, 0.81),  # 20% dead, 30% alive
        0.80: (0.50, 0.54),  # 30% dead, 50% at 54%
    }

    print("=" * 70)
    print("ANALYTIC vs SIMULATION: Critical Mass")
    print("=" * 70)
    print(f"\n{'rho':>6} {'Bound':>10} {'MF(0.75)':>10} {'MF(0.65)':>10} "
          f"{'Sim threshold':>15} {'Sim prod%':>10}")
    print("-" * 65)

    rho_values = sorted(sim_data.keys())
    bounds = []
    mf_75 = []
    mf_65 = []
    sim_thresholds = []

    for rho in rho_values:
        b = analytic_lower_bound(rho)
        m75 = mean_field_critical_mass(rho, c_prod=0.75)
        m65 = mean_field_critical_mass(rho, c_prod=0.65)
        sim_f, sim_p = sim_data[rho]

        bounds.append(b)
        mf_75.append(m75)
        mf_65.append(m65)
        sim_thresholds.append(sim_f)

        print(f"{rho:>6.2f} {b:>10.3f} {m75:>10.3f} {m65:>10.3f} "
              f"{sim_f:>15.0%} {sim_p:>10.0%}")

    print("\nKey:")
    print("  Bound     = rho/(1+rho)  [assumes c_prod = 1]")
    print("  MF(0.75)  = rho/(0.75*(1+rho))  [initial user confidence]")
    print("  MF(0.65)  = rho/(0.65*(1+rho))  [slightly degraded users]")
    print("  Sim threshold = smallest f with >50% producers in simulation")

    if HAS_MPL:
        plot_bounds_comparison(rho_values, bounds, mf_75, mf_65,
                               sim_thresholds)

    return rho_values, bounds, mf_75, mf_65, sim_thresholds


def plot_bounds_comparison(rho_values, bounds, mf_75, mf_65, sim_thresholds):
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(rho_values, bounds, 'o--', color='#3498db', linewidth=2,
            markersize=8, label='Lower bound: f > rho/(1+rho)')
    ax.plot(rho_values, mf_75, 's--', color='#f39c12', linewidth=2,
            markersize=8, label='Mean-field (c_prod=0.75)')
    ax.plot(rho_values, mf_65, '^--', color='#e67e22', linewidth=2,
            markersize=8, label='Mean-field (c_prod=0.65)')
    ax.plot(rho_values, sim_thresholds, 'D-', color='#e74c3c', linewidth=2.5,
            markersize=10, label='Simulation (20 MC runs)')

    # shade the region between bound and simulation
    ax.fill_between(rho_values, bounds, sim_thresholds,
                     alpha=0.1, color='#e74c3c')

    ax.set_xlabel('rho (preemption weight)', fontsize=12)
    ax.set_ylabel('Critical mass (min initial user fraction)', fontsize=12)
    ax.set_title('Analytic Bounds vs Simulation\n'
                 'How well can we predict the tipping point?', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 0.85)
    ax.set_ylim(0, 0.55)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('analytic_vs_simulation.png', dpi=150)
    print(f'\nPlot saved: analytic_vs_simulation.png')
    plt.close()


def plot_mean_field_trajectories():
    """Show mean-field trajectories at different f values for rho=0.3."""
    if not HAS_MPL:
        return

    rho = 0.3
    fractions = [0.02, 0.05, 0.08, 0.10, 0.15, 0.30]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    cmap = plt.cm.viridis

    for i, f in enumerate(fractions):
        color = cmap(i / (len(fractions) - 1))
        traj = mean_field_trajectory(f, rho)
        steps = range(len(traj))

        c_users = [t[0] for t in traj]
        c_nonusers = [t[1] for t in traj]
        p_frac = [t[2] for t in traj]

        ax1.plot(steps, c_nonusers, color=color, linewidth=2,
                 label=f'f={f:.0%}')
        ax2.plot(steps, p_frac, color=color, linewidth=2,
                 label=f'f={f:.0%}')

    ax1.set_xlabel('Time step', fontsize=11)
    ax1.set_ylabel('Non-user mean confidence', fontsize=11)
    ax1.set_title(f'Mean-Field: Non-User Trajectories (rho={rho})', fontsize=12)
    ax1.axhline(0.5, color='gray', ls=':', alpha=0.4)
    ax1.legend(fontsize=9)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.2)

    ax2.set_xlabel('Time step', fontsize=11)
    ax2.set_ylabel('Producer fraction', fontsize=11)
    ax2.set_title(f'Mean-Field: Producer Fraction (rho={rho})', fontsize=12)
    ax2.axhline(0.5, color='gray', ls=':', alpha=0.4)
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('mean_field_trajectories.png', dpi=150)
    print(f'Plot saved: mean_field_trajectories.png')
    plt.close()


if __name__ == '__main__':
    compare_bounds_to_simulation()
    plot_mean_field_trajectories()

    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print("""
The critical mass for a construction to survive is approximately:

    f* ≈ rho / (c_prod * (1 + rho))

where:
    rho    = preemption weight (free parameter)
    c_prod = mean confidence of initial producers (~0.65-0.75)

This gives a closed-form prediction of the tipping point.
To estimate rho from data, observe a real tipping point f* and solve:

    rho = f* * c_prod / (1 - f* * c_prod)

For example, if dialectal forms need ~20% community adoption to
survive (f* ≈ 0.20, c_prod ≈ 0.70):

    rho = 0.20 * 0.70 / (1 - 0.20 * 0.70) = 0.14 / 0.86 ≈ 0.16

This would pin rho empirically.
""")
