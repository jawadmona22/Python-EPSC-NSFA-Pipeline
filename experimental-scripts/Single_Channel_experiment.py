# Comparing stochastic (Gillespie SSA) macroscopic current vs deterministic ODE mean
# for the 3-state model U <-> B <-> O with an agonist pulse between 1 ms and 11 ms.
# Time units: milliseconds (ms). Rates provided by the user are in s^-1, so convert to ms^-1.

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Parameters (converted to per-ms units) ---
kon_s = 1e8  # M^-1 s^-1
koff_s = 5e4  # s^-1
Beta_s = 1e5  # s^-1 (B -> O)
Alpha_s = 1e3  # s^-1 (O -> B)
agonist_conc_M = 1e-3  # 1 mM during pulse

# convert to per-ms (divide by 1000)
kon = kon_s / 1000.0  # M^-1 ms^-1
koff = koff_s / 1000.0  # ms^-1
Beta = Beta_s / 1000.0  # ms^-1
Alpha = Alpha_s / 1000.0  # ms^-1

# Effective forward rate when agonist present: k1 = kon * [A]
k1_on_present = kon * agonist_conc_M  # ms^-1

# Channel/cell parameters
N = 500
g = 20e-12  # S (20 pS)
V_hold = -100e-3
V_rev = 0.0

# Pulse definition (ms)
pulse_start = 1.0  # ms
pulse_end = 11.0  # ms

# Simulation time grid (ms)
t_end = 50.0  # ms
n_samples = 2000
t_eval = np.linspace(0.0, t_end, n_samples)  # ms


# --- Gillespie SSA for one channel ---
def simulate_one_channel_gillespie(t_max_ms, seed=None):
    rng = np.random.default_rng(seed)
    t = 0.0
    state = 0  # 0=U, 1=B, 2=O
    times = [t]
    states = [state]
    while t < t_max_ms:
        # compute propensities (per-ms)
        if state == 0:
            rate_ub = k1_on_present if (pulse_start <= t <= pulse_end) else 0.0
            propensities = [(1, rate_ub)]
        elif state == 1:
            propensities = [(0, koff), (2, Beta)]
        elif state == 2:
            propensities = [(1, Alpha)]
        else:
            break

        rates = np.array([r for (_, r) in propensities])
        a0 = rates.sum()
        # next pulse boundary where rates change
        boundaries = []
        if t < pulse_start:
            boundaries.append(pulse_start)
        if t < pulse_end:
            boundaries.append(pulse_end)
        boundary = min(boundaries) if boundaries else np.inf

        if a0 <= 0:
            # no reactions possible until boundary; jump to boundary or t_max
            t = min(boundary, t_max_ms)
            times.append(t)
            states.append(state)
            if t >= t_max_ms:
                break
            continue

        # sample waiting time (ms)
        tau = rng.exponential(1.0 / a0)
        if t + tau > min(boundary, t_max_ms):
            # advance to boundary without reaction
            t = min(boundary, t_max_ms)
            times.append(t)
            states.append(state)
            continue

        # reaction occurs
        t = t + tau
        # choose reaction by weight
        r = rng.uniform(0, a0)
        cum = 0.0
        chosen_state = state
        for (target, rate) in propensities:
            cum += rate
            if r <= cum:
                chosen_state = target
                break
        state = chosen_state
        times.append(t)
        states.append(state)

    # ensure we have an event at t_max
    if times[-1] < t_max_ms:
        times.append(t_max_ms)
        states.append(states[-1])
    return np.array(times), np.array(states, dtype=int)


# convert event list to sampled open trace at t_eval
def events_to_sampled_open(times, states, t_eval):
    sampled = np.zeros_like(t_eval, dtype=int)
    idx = 0
    for i, tt in enumerate(t_eval):
        while idx + 1 < len(times) and times[idx + 1] <= tt:
            idx += 1
        sampled[i] = 1 if states[idx] == 2 else 0
    return sampled


# --- Simulate population (stochastic) ---
np.random.seed(42)
open_counts = np.zeros_like(t_eval, dtype=int)
for n in range(N):
    times_n, states_n = simulate_one_channel_gillespie(t_end, seed=np.random.randint(0, 2 ** 31 - 1))
    open_counts += events_to_sampled_open(times_n, states_n, t_eval)

I_stoch = open_counts * g * (V_hold - V_rev)  # A
I_stoch_pA = I_stoch * 1e12


# --- Deterministic mean-field via solve_ivp ---
def dPdt_ms(t, P):
    # P = [P_U, P_B, P_O]
    k1 = k1_on_present if (pulse_start <= t <= pulse_end) else 0.0
    Q = np.array([
        [-k1, koff, 0.0],
        [k1, -(koff + Beta), Alpha],
        [0.0, Beta, -Alpha]
    ])
    return Q.dot(P)


P0 = np.array([1.0, 0.0, 0.0])
sol = solve_ivp(dPdt_ms, (0.0, t_end), P0, t_eval=t_eval, method='BDF', rtol=1e-8, atol=1e-12)
P_O_det = sol.y[2]  # open probability over time
I_det_pA = (N * g * P_O_det * (V_hold - V_rev)) * 1e12

# --- Plotting ---
plt.figure(figsize=(9, 3))
plt.plot(t_eval, I_stoch_pA)
plt.xlabel('Time (ms)')
plt.ylabel('Current (pA)')
plt.title(f'Stochastic macroscopic current, N={N} (one trial)')
plt.axvspan(pulse_start, pulse_end, alpha=0.15)
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 3))
plt.plot(t_eval, I_det_pA)
plt.xlabel('Time (ms)')
plt.ylabel('Current (pA)')
plt.title('Deterministic mean-field current (ODE)')
plt.axvspan(pulse_start, pulse_end, alpha=0.15)
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 3))
plt.plot(t_eval, I_stoch_pA, label='Stochastic (one trial)')
plt.plot(t_eval, I_det_pA, label='Deterministic (mean-field)', linewidth=1)
plt.xlabel('Time (ms)')
plt.ylabel('Current (pA)')
plt.title('Stochastic vs Deterministic macroscopic current')
plt.legend()
plt.axvspan(pulse_start, pulse_end, alpha=0.12)
plt.tight_layout()
plt.show()

# Print summary
steady_open_fraction_det = P_O_det[-1]
I_ss_det_pA = (N * g * steady_open_fraction_det * (V_hold - V_rev)) * 1e12
print(f"Deterministic open fraction at t={t_end} ms: {steady_open_fraction_det:.6e}")
print(f"Deterministic current at t={t_end} ms: {I_ss_det_pA:.4f} pA")

