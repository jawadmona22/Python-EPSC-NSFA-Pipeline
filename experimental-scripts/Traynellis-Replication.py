# Re-run the Gillespie simulation and plotting for the 3-state model.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp


# Parameters
kon = 1e8 #s-1
koff = 5e4
Beta = 1e5
Alpha = 1e3
agonist_conc = 1e-3

N = 50
g = 20e-12
V_hold = -100e-3
V_rev = 0.0

pulse_start = 0.001
pulse_end = pulse_start + .000025  #One time step

t_max = 0.06
n_samples = 2000
t_eval = np.linspace(0, t_max, n_samples)
time_ms = t_eval * 1e3

num_traces = 100

def agonist_conc_t(t):
    return agonist_conc if (pulse_start <= t <= pulse_end) else 0.0

def rates_for_state(state, t):
    if state == 0:
        return [(1, kon * agonist_conc_t(t))]
    elif state == 1:
        return [(0, koff), (2, Beta)]
    elif state == 2:
        return [(1, Alpha)]
    else:
        raise ValueError

def next_pulse_boundary_after(t):
    boundaries = []
    if t < pulse_start:
        boundaries.append(pulse_start)
    if t < pulse_end:
        boundaries.append(pulse_end)
    return min(boundaries) if boundaries else np.inf

def simulate_one_channel(t_max):
    t = 0.0
    state = 0
    times = [t]
    states = [state]
    while t < t_max:
        trans = rates_for_state(state, t) #[(1,0)] = trans
        rates = np.array([r for (_, r) in trans])
        a0 = rates.sum()
        boundary = next_pulse_boundary_after(t)
        if a0 <= 0:
            t = min(boundary, t_max)
            times.append(t)
            states.append(state)
            if t >= t_max:
                break
            continue
        dt = -np.log(np.random.rand()) / a0
        if t + dt > min(boundary, t_max):
            t = min(boundary, t_max)
            times.append(t)
            states.append(state)
            continue
        t = t + dt
        r = np.random.rand() * a0
        cum = 0.0
        chosen = None
        for i, (_, rate) in enumerate(trans):
            cum += rate
            if r <= cum:
                chosen = trans[i][0]
                break
        if chosen is None:
            chosen = trans[-1][0]
        state = chosen
        times.append(t)
        states.append(state)
    if times[-1] < t_max:
        times.append(t_max)
        states.append(states[-1])
    return np.array(times), np.array(states, int)

def events_to_sampled_open(times, states, t_eval):
    sampled = np.zeros_like(t_eval, dtype=int)
    idx = 0
    for i, tt in enumerate(t_eval):
        while idx + 1 < len(times) and times[idx + 1] <= tt:
            idx += 1
        sampled[i] = 1 if states[idx] == 2 else 0
    return sampled

# Single-channel example
np.random.seed(1)
times1, states1 = simulate_one_channel(t_max)
open_trace1 = events_to_sampled_open(times1, states1, t_eval)

plt.figure(figsize=(10,2.5))
plt.step(time_ms, open_trace1, where='post')
plt.ylim(-0.1, 1.1)
plt.xlabel('Time (ms)')
plt.ylabel('Open (1) / Closed (0)')
plt.title('Single-channel stochastic trace (Gillespie SSA)')
plt.tight_layout()
plt.show()

# Simulate N channels
open_counts = np.zeros_like(t_eval, dtype=int)
for n in range(N):
    times_n, states_n = simulate_one_channel(t_max)
    open_counts += events_to_sampled_open(times_n, states_n, t_eval)

I = open_counts * g * (V_hold - V_rev)
I_pA = I * 1e12

plt.figure(figsize=(10,3))
plt.plot(time_ms, I_pA)
plt.xlabel('Time (ms)')
plt.ylabel('Current (pA)')
plt.title(f'Macroscopic current from N={N} stochastic channels (Gillespie SSA)')
plt.axvspan(pulse_start*1e3, pulse_end*1e3, alpha=0.2)
plt.tight_layout()
plt.show()

# Deterministic mean-field for comparison
def dPdt(t, P):
    k1 = kon * agonist_conc if (pulse_start <= t <= pulse_end) else 0.0
    Q = np.array([
        [-k1,       koff,      0.0],
        [ k1, -(koff + Beta), Alpha],
        [ 0.0,      Beta,   -Alpha]
    ])
    return Q @ P

P0 = np.array([1.0, 0.0, 0.0])
sol = solve_ivp(dPdt, (0, t_max), P0, t_eval=t_eval, method='BDF', rtol=1e-8, atol=1e-12)
P_O = sol.y[2]
I_det_pA = (N * g * P_O * (V_hold - V_rev)) * 1e12

plt.figure(figsize=(10,3))
plt.plot(time_ms, I_pA, label='Stochastic (single trial)')
plt.plot(time_ms, I_det_pA, label='Deterministic (mean-field)', linewidth=1)
plt.xlabel('Time (ms)')
plt.ylabel('Current (pA)')
plt.title('Stochastic (one trial) vs Deterministic mean current')
plt.legend()
plt.axvspan(pulse_start*1e3, pulse_end*1e3, alpha=0.15)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10,3))
plt.plot(time_ms, I_pA)
plt.plot(time_ms, I_det_pA, label='Deterministic (mean-field)', linewidth=1)
plt.xlabel('Time (ms)')
plt.ylabel('Current (pA)')
plt.title('Stochastic (one trial) vs Deterministic mean current')
plt.legend()
plt.axvspan(pulse_start*1e3, pulse_end*1e3, alpha=0.15)
plt.tight_layout()
plt.show()

steady_open_fraction_det = P_O[-1]
I_ss_det_pA = (N * g * steady_open_fraction_det * (V_hold - V_rev)) * 1e12
print(f"Deterministic steady-state open fraction at end of simulation: {steady_open_fraction_det:.6f}")
print(f"Deterministic steady-state current: {I_ss_det_pA:.2f} pA")

# Simulate EPSCs
all_traces = []

for trace_id in range(num_traces):
    open_counts = np.zeros_like(t_eval, dtype=int)

    # simulate N independent channels and sum their openings
    for n in range(N):
        times_n, states_n = simulate_one_channel(t_max)
        open_counts += events_to_sampled_open(times_n, states_n, t_eval)

    # convert to current
    I = open_counts * g * (V_hold - V_rev)  # Amps
    I_pA = I * 1e12                         # picoAmps

    all_traces.append(I_pA)

# Convert to DataFrame: each column is one trace, rows are timepoints
df = pd.DataFrame(np.array(all_traces).T, index=t_eval)

plt.xlabel("Time (ms)")
plt.ylabel("Current (pA)")
for col in df.columns:
    plt.plot(df.index*1e3,df[col],alpha=.3)
template = np.mean(df,axis=1)
plt.plot(df.index*1e3,template)
plt.title("Traynelis Replica, 100 EPSCs")
plt.show()
df.columns = [f"trace_{i+1}" for i in range(num_traces)]


df = df*-1
df.to_excel("Tranyelis_replica_EPSCs.xlsx")
