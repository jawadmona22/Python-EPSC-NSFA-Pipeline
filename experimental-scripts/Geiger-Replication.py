# Re-run the Gillespie simulation and plotting for the more complex Geiger model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
import math
from tqdm import tqdm
# Rate Constants in s-1 unless indicated as m-1s-1
kC0C1 = 17.1e6 #m-1s-1
kC1C0 = 157
kC1C2 = 3.24e6 #m-1s-1
kC2C1 = 3.76e3
kC2O = 14.9e3
kOC2 = 4e3 * 3
kC1C3 = 1.53e3
kC3C1 = 408
kC2C4 = 502
kC4C2 = .377
kOC5 = 121 * 3
kC5O = 191
kC3C4 = .611e6 #m-1s-1
kC4C3 = 2
kC4C5 = 1.59e3
kC5C4 = 899e3


#State definitions
STATE_MAP = {'C0':0, 'C1':1, 'C2':2, 'C3':3, 'C4':4, 'C5':5, 'O':6}
STATE_NAMES = {v:k for k,v in STATE_MAP.items()}

N = 50 #num channels
g = 8.5e-12 #conductance
V_hold = -60e-3 #holding potential
V_rev = 0  #reversal potential

##Agonist Parameters
agonist_conc = 1e-3 * 100
pulse_start = 0 #Start at beginning
pulse_end = 0.001 #end at 1ms


##Time Parameters
t_max = 0.01 # 10ms
n_samples = 1000 #Time interval defined as 10microseconds, across 10ms, means 10,000microseconds and 1000 samples
t_eval = np.linspace(0,t_max,n_samples)
time_ms = t_eval * 1e3

num_traces = 1000


def JGT(t): #scale in mM
    """
    Returns glutamate concentration (in M) at time t [seconds].
    """
    idx = np.abs(t_eval-t).argmin()
    if idx < 5:
        return 0
    t_corrected = (idx-5) * 0.00001 if idx < 34 else 0
    conc_1 = (1 - math.exp(-0.00001 / 0.000000002)) * math.exp(-0.00001 /  0.00002)
    # Bi-exponential function
    conc = (1 - math.exp(-t_corrected / 0.000000002)) * math.exp(-t_corrected /  0.00002)

    # Scale to ~1 mM peak (in M)
    conc = conc * (agonist_conc/conc_1)


    return conc
#
t = np.linspace(0, .01, 1000)

glutamate = [JGT(tt) for tt in t]
print(glutamate)
plt.plot(t * 1e3, glutamate,color='red',label='JGT',alpha=1)  # convert to ms for x-axis
veruki_transient = np.zeros(1000)
for i in range(0,100):
    veruki_transient[i] = 1e-3
plt.plot(t*1e3,veruki_transient,color='blue',label='Veruki',alpha=.5)
plt.legend()
plt.xlabel("Time (ms)")
plt.ylabel("[Glutamate] (M)")
plt.title("Glu Transients (1 mM peak)")
plt.show()


def agonist_conc_t(t,type='JGT'):
    if type == 'JGT':
        return JGT(t)
    if (pulse_start <= t <= pulse_end):
        return agonist_conc
    else:
        return 0.0



def rates_for_state(state, t):
    if state == 0:
        return [(1, kC0C1 * agonist_conc_t(t,type='JGT'))]
    elif state == 1:
        return [(0, kC1C0), (3, kC1C3),(2,kC1C2 * agonist_conc_t(t,type='JGT'))]
    elif state == 2:
        return [(1, kC2C1),(4,kC2C4),(6,kC2O)]
    elif state == 3:
        return [(1, kC3C1), (4, kC3C4 * agonist_conc_t(t,type='JGT'))]
    elif state == 4:
        return [(2, kC4C2), (5, kC4C5),(3,kC4C3)]
    elif state == 5:
        return [(6, kC5O), (4, kC5C4)]
    elif state == 6:
        return [(2, kOC2), (5, kOC5)]
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
        # boundary = next_pulse_boundary_after(t)
        if a0 <= 0:
            # t = min(boundary, t_max)
            t += .00001
            times.append(t)
            states.append(state)
            if t >= t_max:
                break
            continue
        dt = -np.log(np.random.rand()) / a0
        # if t + dt > min(boundary, t_max):
        #     t = min(boundary, t_max)
        #     times.append(t)
        #     states.append(state)
        #     continue
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
        while idx + 1 < len(times) and times[idx + 1] <= tt:  #Converts the dt to the sample rate we want
            idx += 1
        sampled[i] = 1 if states[idx] == 'O' else 0
    return sampled


def pick_number_channels(method="uniform", sd=0, mean=0, mog_params=None):
    lower_limit = 50
    upper_limit = 4000

    if method == "uniform":
        num_channels = np.random.uniform(lower_limit, upper_limit)

    elif method == "normal":
        #mean = 1100  # Center of the range
        #std_dev = 300  # Ensures 99.7% of the data is between 200 and 2000
        num_channels = int(np.clip(np.random.normal(mean, sd), lower_limit, upper_limit))

    elif method == "lognormal":
        num_channels = int(np.clip(np.random.lognormal(mean, sd), lower_limit, upper_limit))

    elif method == "mog":  # Mixture of Gaussians
        if mog_params is None:
            raise ValueError("mog_params must be provided for Mixture of Gaussians")

        num_components = len(mog_params["means"])
        weights = np.array(mog_params["weights"])
        weights /= weights.sum()  # Ensure weights sum to 1

        # Pick a Gaussian component based on weights
        component = np.random.choice(num_components, p=weights)
        mean = mog_params["means"][component]
        std_dev = mog_params["std_devs"][component]

        # Sample from the selected Gaussian and clip
        num_channels = int(np.clip(np.random.normal(mean, std_dev), lower_limit, upper_limit))

    return num_channels





# # Single-channel example
# np.random.seed(1)
# times1, states1 = simulate_one_channel(t_max)
# open_trace1 = events_to_sampled_open(times1, states1, t_eval)
#
# plt.figure(figsize=(10,2.5))
# plt.step(time_ms, open_trace1, where='post')
# plt.ylim(-0.1, 1.1)
# plt.xlabel('Time (ms)')
# plt.ylabel('Open (1) / Closed (0)')
# plt.title('Single-channel stochastic trace (Gillespie SSA)')
# plt.tight_layout()
# plt.show()

# # Simulate N channels
# open_counts = np.zeros_like(t_eval, dtype=int)
# for n in range(N):
#     times_n, states_n = simulate_one_channel(t_max)
#     open_counts += events_to_sampled_open(times_n, states_n, t_eval)
#
# I = open_counts * g * (V_hold - V_rev)
# I_pA = I * 1e12

# Simulate EPSCs
all_traces = []

for trace_id in tqdm(range(num_traces), desc="Processing traces"):
    open_counts = np.zeros_like(t_eval, dtype=int)

    # simulate N independent channels and sum their openings
    N = pick_number_channels(method="normal",mean=2200,sd=700)
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
plt.title("Veruki GluGei99 Replica, 1000 EPSCs")
plt.show()
df.columns = [f"trace_{i+1}" for i in range(num_traces)]


df = df*-1
df.to_excel(f"experimental-scripts/Changing_Geiger/Geiger_replica_EPSCs_{str(int(agonist_conc*1000))}mM_JGT_3x_n.xlsx")