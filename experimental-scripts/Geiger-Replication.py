# Re-run the Gillespie simulation and plotting for the more complex Geiger model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp


# Rate Constants
kC0C1 = 17.1e6 #m-1s-1
kC1C0 = 157
kC1C2 = 3.24e6
kC2C1 = 3.76e3
kC2O = 14.9e3
kOC2 = 4e3
kC1C3 = 1.53e3
kC3C1 = 408
kC2C4 = 502
kC4C2 = .377
kOC5 = 121
kC5O = 191
kC3C4 = .611e6
kC4C3 = 2
kC4C5 = 1.59e3
kC5C4 = 899e3





N = 50 #num channels
g = 8.5e-12 #conductance
V_hold = -60e-3 #holding potential
V_rev = 0  #reversal potential

##Agonist Parameters
agonist_conc = 1e-3
pulse_start = 0.001 #Start at 1 ms
pulse_end = pulse_start + .00001 #1 time step of 10microseconds


##Time Parameters
t_max = 0.01 # 10ms
n_samples = 1000 #Time interval defined as 10microseconds, across 10ms, means 10,000microseconds and 1000 samples
t_eval = np.linspace(0,t_max,n_samples)
time_ms = t_eval * 1e3

num_traces = 100

def agonist_conc_t(t):
    return agonist_conc if (pulse_start <= t <= pulse_end) else 0.0


def rates_for_state(state, t):
    if state == 'C0':
        return [('C1', kC0C1 * agonist_conc_t(t))]
    elif state == 'C1':
        return [('C0', kC1C0), ('C3', kC1C3),('C2',kC1C2)]
    elif state == 'C2':
        return [('C1', kC2C1),('C4',kC2C4),('O',kC2O)]
    elif state == 'C3':
        return [('C1', kC3C1), ('C4', kC3C4)]
    elif state == 'C4':
        return [('C2', kC4C2), ('C5', kC4C5),('C3',kC4C3)]
    elif state == 'C5':
        return [('O', kC5O), ('C4', kC5C4)]
    elif state == 'O':
        return [('C2', kOC2), ('C5', kOC5)]
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
    state = 'C0'
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
    return np.array(times), np.array(states, str)


def events_to_sampled_open(times, states, t_eval):
    sampled = np.zeros_like(t_eval, dtype=int)
    idx = 0
    for i, tt in enumerate(t_eval):
        while idx + 1 < len(times) and times[idx + 1] <= tt:
            idx += 1
        sampled[i] = 1 if states[idx] == 'O' else 0
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
df.to_excel("Geiger_replica_EPSCs.xlsx")