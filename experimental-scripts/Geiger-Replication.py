# Re-run the Gillespie simulation and plotting for the more complex Geiger model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
import math
from tqdm import tqdm

params = {
    'fixed_glu_conc': 1, #mM
    'glu_distribution':'normal' ,#fixed or normal
    'glu_mean':None,
    'glu_std':None,
    'channel_distribution':'normal', #fixed or normal
    'transient_type': 'Veruki', #Veruki or JGT
    'RC_multiplier': 3,
    'channel_mean': 2200,
    'channel_std':700,
    'fixed_n_value':None,
    'channels_lower_lim': 0,
    'ci_ratio':0,
    'ci_current': -.51 ,#in picoamps
    'cp_current': -1.53


}

num_traces = 1000



# Rate Constants in s-1 unless indicated as m-1s-1
kC0C1 = 17.1e6 #m-1s-1
kC1C0 = 157
kC1C2 = 3.24e6 #m-1s-1
kC2C1 = 3.76e3
kC2O = 14.9e3
kOC2 = 4e3 * params['RC_multiplier']
kC1C3 = 1.53e3
kC3C1 = 408
kC2C4 = 502
kC4C2 = .377
kOC5 = 121 * params['RC_multiplier']
kC5O = 191
kC3C4 = .611e6 #m-1s-1
kC4C3 = 2
kC4C5 = 1.59e3
kC5C4 = 899e3


#State definitions
STATE_MAP = {'C0':0, 'C1':1, 'C2':2, 'C3':3, 'C4':4, 'C5':5, 'O':6}
STATE_NAMES = {v:k for k,v in STATE_MAP.items()}


#########
g = 8.5e-12 #conductance
V_hold = -60e-3 #holding potential
V_rev = 0  #reversal potential

#########


##Agonist Parameters
# agonist_conc = 1e-3 * params['fixed_glu_conc']
pulse_start = 0 #Start at beginning
pulse_end = 0.001 #end at 1ms


##Time Parameters
t_max = 0.01 # 10ms
n_samples = 1000 #Time interval defined as 10microseconds, across 10ms, means 10,000microseconds and 1000 samples
t_eval = np.linspace(0,t_max,n_samples)
time_ms = t_eval * 1e3



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
# t = np.linspace(0, .01, 1000)
#
# glutamate = [JGT(tt) for tt in t]
# print(glutamate)
# plt.plot(t * 1e3, glutamate,color='red',label='JGT',alpha=1)  # convert to ms for x-axis
# veruki_transient = np.zeros(1000)
# for i in range(0,100):
#     veruki_transient[i] = 1e-3
# plt.plot(t*1e3,veruki_transient,color='blue',label='Veruki',alpha=.5)
# plt.legend()
# plt.xlabel("Time (ms)")
# plt.ylabel("[Glutamate] (M)")
# plt.title("Glu Transients (1 mM peak)")
# plt.show()



def agonist_conc_t(t):
    type = params['transient_type']
    if type == 'JGT':
        return JGT(t)
    if type == 'Veruki':
        if (pulse_start <= t <= pulse_end):
            return agonist_conc
        else:
            return 0.0
    else:
        print("WARNING! Invalid agonist type")




# def rates_for_state(state, t):
#     if state == 0:
#         return [(1, kC0C1 * agonist_conc_t(t,type=params['transient_type']))]
#     elif state == 1:
#         return [(0, kC1C0), (3, kC1C3),(2,kC1C2 * agonist_conc_t(t,type=params['transient_type']))]
#     elif state == 2:
#         return [(1, kC2C1),(4,kC2C4),(6,kC2O)]
#     elif state == 3:
#         return [(1, kC3C1), (4, kC3C4 * agonist_conc_t(t,type=params['transient_type']))]
#     elif state == 4:
#         return [(2, kC4C2), (5, kC4C5),(3,kC4C3)]
#     elif state == 5:
#         return [(6, kC5O), (4, kC5C4)]
#     elif state == 6:
#         return [(2, kOC2), (5, kOC5)]
#     else:
#         raise ValueError

def next_pulse_boundary_after(t):
    boundaries = []
    if t < pulse_start:
        boundaries.append(pulse_start)
    if t < pulse_end:
        boundaries.append(pulse_end)
    return min(boundaries) if boundaries else np.inf

def simulate_one_channel(t_max,agonist_conc):
    t = 0.0
    state = 0
    times = [t]
    states = [state]
    while t < t_max:
        rates = RATES[state](t)
        dests = DESTS[state]
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
        for i, rate in enumerate(rates):
            cum += rate
            if r <= cum:
                state = dests[i]
                break

        times.append(t)
        states.append(state)
    if times[-1] < t_max:
        times.append(t_max)
        states.append(states[-1])
    return np.array(times), np.array(states)


def events_to_sampled_open(times, states, t_eval):
    sampled = np.zeros_like(t_eval, dtype=int)
    idx = 0
    for i, tt in enumerate(t_eval):
        while idx + 1 < len(times) and times[idx + 1] <= tt:  #Converts the dt to the sample rate we want
            idx += 1
        sampled[i] = 1 if states[idx] == 6 else 0 #6 == O here
    return sampled





# Simulate EPSCs
RATES = {
    0: lambda t: np.array([kC0C1 * agonist_conc_t(t)]),
    1: lambda t: np.array([kC1C0, kC1C3, kC1C2 * agonist_conc_t(t)]),
    2: lambda t: np.array([kC2C1, kC2C4, kC2O]),
    3: lambda t: np.array([kC3C1, kC3C4 * agonist_conc_t(t)]),
    4: lambda t: np.array([kC4C2, kC4C5, kC4C3]),
    5: lambda t: np.array([kC5O, kC5C4]),
    6: lambda t: np.array([kOC2, kOC5])
}
DESTS = {
    0: np.array([1]),           # C0 → C1
    1: np.array([0, 3, 2]),     # C1 → C0, C3, C2
    2: np.array([1, 4, 6]),     # C2 → C1, C4, O
    3: np.array([1, 4]),        # C3 → C1, C4
    4: np.array([2, 5, 3]),     # C4 → C2, C5, C3
    5: np.array([6, 4]),        # C5 → O, C4
    6: np.array([2, 5])         # O → C2, C5
}



ratio_list = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]
for ci_ratio in ratio_list:
    params['ci_ratio'] = ci_ratio
    all_traces = []
    np.random.seed(189)
    if params['channel_distribution'] == 'normal':
        mean = params['channel_mean']
        std = params['channel_std']
        lower_lim = params['channels_lower_lim']
        N_array = np.random.normal(mean, std, num_traces)
        N_array = np.clip(N_array, lower_lim, 4000)



    elif params['channel_distribution'] == 'fixed':
        fixed_N = params['fixed_n_value']
        N_array =  np.full(num_traces, fixed_N)


    if params['glu_distribution'] == 'normal':
        mean = params['glu_mean']
        std = params['glu_std']
        glu_array = np.random.normal(2.5,1,num_traces)

    elif params['glu_distribution'] == 'fixed':
        fixed_glu = params['fixed_glu_conc']
        glu_array = np.full(num_traces,fixed_glu)

    ci_threshold = num_traces * params['ci_ratio'] #at this threshold we switch to the CP amplitude
    ci_multiple = int(params['ci_ratio']  *10)
    for trace_id in tqdm(range(num_traces), desc="Processing traces"):
        open_counts = np.zeros_like(t_eval, dtype=int)

        # simulate N independent channels and sum their openings
        N = int(N_array[trace_id])
        agonist_conc = glu_array[trace_id] * 1e-3

        for n in range(N):
            times_n, states_n = simulate_one_channel(t_max,agonist_conc)
            open_counts += events_to_sampled_open(times_n, states_n, t_eval)

        # convert to current
        if trace_id <= ci_threshold:

            I_pA = open_counts * params['ci_current']#g * (V_hold - V_rev)  # Amps
        else:
            I_pA = open_counts * params['cp_current']#g * (V_hold - V_rev)  # Amps


        #I_pA = I * 1e12                         # picoAmps

        all_traces.append(I_pA)

    # Convert to DataFrame: each column is one trace, rows are timepoints
    df = pd.DataFrame(np.array(all_traces).T, index=t_eval)

    # plt.xlabel("Time (ms)")
    # plt.ylabel("Current (pA)")
    # for col in df.columns:
    #     plt.plot(df.index*1e3,df[col],alpha=.3)
    # template = np.mean(df,axis=1)
    # plt.plot(df.index*1e3,template)
    # plt.title("GluGei99")
    # plt.show()
    df.columns = [f"trace_{i+1}" for i in range(num_traces)]


    df = df*-1
    type = params['transient_type'][0:3]
    mult = str(params['RC_multiplier'])
    dist = params['channel_distribution'][0]
    glu_dist = params['glu_distribution'][0]
    df.to_excel(f"C:/Users/j.mona/Documents/GitHub/Python-EPSC-NSFA-Pipeline/experimental-scripts//CI_CP_Redo/normalglu_normaln_ci{ci_multiple}.xlsx")