import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
def Agonist_Pulse(glutamate_scale = 1,steady=False,second_pulse=False):
    AgPulse = np.zeros((2000))
    if steady == True:
        for i in range (10,2000):
            AgPulse[i] = 1000 * glutamate_scale

    else:
        for i in range (10,1000):
            t = (i-9) * 0.00002
            AgPulse[i] = (5 * (1 - math.exp(-t / 0.000000002))) * math.exp(-t / 0.00002)
            AgPulse[i] = 1000 * glutamate_scale* AgPulse[i] #This makes the peak 1839.3972058572117 micro molar (1.839 mM)
            AgPulse[i] = AgPulse[i]/1.8393972058572117 #Now we have 1.0 mM
            if AgPulse[i] < .001:
                AgPulse[i] = 0
        if second_pulse:
            AgPulse[100:1090] = AgPulse[10:1000]
    return AgPulse



def StateC0(npl,AgPulse):
    while AgPulse[npl] == 0:
        npl = npl + 1
        if npl > 800:
            return npl, "End"
    #AgPulse not 0
    return npl, "StateC1"

def StateC1(npl,lotbl,AgPulse):
    tpoint = .00002
    kC1C0 = 157 * tpoint
    kC1C2 = 3240000 * tpoint
    kC1C3 = 1530 * tpoint
    if npl > 800:
        return npl, "End"
    d10 = (1 / (3000 * kC1C0)) * lotbl[int(10000*random.uniform(0,1))]
    if AgPulse[npl] == 0:
        d12 = 900000
    else:
        d12 = (1 / (3000 * 0.000001 * AgPulse[npl] * kC1C2)) *lotbl[int(10000*random.uniform(0,1))]
    d13 = (1 / (3000 * kC1C3)) * lotbl[int(10000*random.uniform(0,1))]
    if (d10 < d12) and (d10 < d13):
        d10 = math.ceil(.5 + d10)
        npl = npl + d10
        return npl, "StateC0"

    if (d13 < d10) and (d13 < d12):
        d13 = math.ceil(.5 + d13)
        npl = npl + d13
        return npl, "StateC3"

    if (d12 < d10) and (d12 < d13):
        d12 = math.ceil(.5 + d12)
        npl = npl + d12
        return npl, "StateC2"

    return npl,"StateC0"

def StateC2(npl,lotbl):
    tpoint = .00002
    kC2C1 = 3760 * tpoint
    kC2O = 14900 * tpoint
    kC2C4 = 502 * tpoint
    if npl > 800:
        return npl, "End"

    d21 = (1 / (3000 * kC2C1)) * lotbl[int(10000*random.uniform(0,1))]
    d24 = (1 / (3000 * kC2C4)) * lotbl[int(10000*random.uniform(0,1))]
    d2O = (1 / (3000 * kC2O)) * lotbl[int(10000*random.uniform(0,1))]

    if (d21 < d24) and (d21 < d2O):
        d21 = math.ceil(.5 + d21)
        npl = npl + d21
        return npl, "StateC1"

    if (d24 < d21) and (d24 < d2O):
        d24 = math.ceil(.5 + d24)
        npl = npl + d24
        return npl, "StateC4"

    if (d2O < d21) and (d2O < d24):
        d2O = math.ceil(.5 + d2O)
        npl = npl + d2O
        return npl, "Open"

    return npl, "StateC1"


def StateC3(npl,lotbl,AgPulse):
    tpoint = .00002
    kC3C1 = 408 * tpoint
    kC3C4 = 611000 * tpoint
    if npl > 800:
        return npl, "End"
    d31 = (1 / (3000 * kC3C1)) * lotbl[int(10000*random.uniform(0,1))]
    if AgPulse[npl] == 0:
        d34 = 900000
    else:
        d34 = (1 / (3000 * 0.000001 * AgPulse[npl] * kC3C4)) *lotbl[int(10000*random.uniform(0,1))]
    if (d31 < d34):
        d31 = math.ceil(.5 + d31)
        npl = npl + d31
        return npl, "StateC1"

    if(d34 <= d31):
        d34 = math.ceil(.5 + d34)
        npl = npl + d34
        return npl, "StateC4"

def StateC4(npl,lotbl):
    if npl > 800:
        return npl, "End"
    tpoint = .00002
    kC4C3 = 2 * tpoint
    kC4C2 = 0.377 * tpoint
    kC4C5 = 1590 * tpoint
    d43 = (1 / (3000 * kC4C3)) * lotbl[int(10000*random.uniform(0,1))]
    d42 = (1 / (3000 * kC4C2)) * lotbl[int(10000*random.uniform(0,1))]
    d45 = (1 / (3000 * kC4C5)) * lotbl[int(10000*random.uniform(0,1))]

    if (d43 < d42) and (d43 < d45):
        d43 = math.ceil(.5 + d43)
        npl = npl + d43
        return npl, "StateC3"

    if (d42 < d43) and (d42 < d45):
        d42 = math.ceil(.5 + d42)
        npl = npl + d42
        return npl, "StateC2"

    if (d45 < d43) and (d45 < d42):
        d45 = math.ceil(.5 + d45)
        npl = npl + d45
        return npl, "StateC5"

    else:
        return npl,"StateC3"

def StateC5(npl,lotbl):
    if npl > 800:
        return npl, "End"
    tpoint = .00002
    kC5C4 = 899000 * tpoint
    kC5O = 191 * tpoint
    d54 = (1 / (3000 * kC5C4)) * lotbl[int(10000*random.uniform(0,1))]
    d5O = (1 / (3000 * kC5O)) * lotbl[int(10000*random.uniform(0,1))]

    if (d54 < d5O):
        d54 = math.ceil(.5 + d54)
        npl = npl + d54
        return npl, "StateC4"

    if (d5O <= d54):
        d5O = math.ceil(.5 + d5O)
        npl = npl + d5O
        return npl, "Open"

def Open(npl,lotbl):
    if npl > 800:
        return npl, "End"
    tpoint = .00002
    kOC2 = 12000 * tpoint #8000 * tpoint
    kOC5 = 363 * tpoint #242 * tpoint
    dO5 = (1 / (3000 * kOC5)) * lotbl[int(10000*random.uniform(0,1))]
    dO2 = (1 / (3000 * kOC2)) * lotbl[int(10000*random.uniform(0,1))]

    if (dO5 < dO2):
        dO5 = math.ceil(.5 + dO5)
        npl = npl + dO5
        return npl, "StateC5"

    if (dO2 <= dO5):
        dO2 = math.ceil(.5 + dO2)
        npl = npl + dO2
        return npl, "StateC2"


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

def pick_glutamate_scale(glutamate_params, method="normal", mog_params=None): #scale in mM
    mean = glutamate_params["gl_mean"]
    sd = glutamate_params["gl_sd"]
    method = glutamate_params["distribution_type"][0]
    continuous = glutamate_params['continuous'][0] if 'continuous' in glutamate_params.columns else None
    lower_limit = 1
    upper_limit = 10

    if method == "uniform":
        glutamate_scale = np.random.uniform(glutamate_params["lower_limit"], glutamate_params["upper_limit"])

    elif method == "normal":


        if continuous:
            glutamate_scale = np.clip(np.random.normal(mean, sd), lower_limit, upper_limit)

        else:
            glutamate_scale = int(np.clip(np.random.normal(mean, sd), lower_limit, upper_limit))

    elif method == "lognormal":
        if continuous:
            glutamate_scale = np.random.lognormal(mean, sd)

        else:
            glutamate_scale = int(np.clip(np.random.lognormal(mean, sd), lower_limit, upper_limit))

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
        #num_channels = int(np.clip(np.random.normal(mean, std_dev), lower_limit, upper_limit))




    return glutamate_scale



def EPSC_Calc(num_EPSCs,channel_params,glutamate_params=[],mog_params = [],current_params = [],output_file_path=None,folder_path = None,agonist_steady=False,double_agonist=False):
    channel_distribution = channel_params["distribution_type"][0]
    glut_distribution = glutamate_params["distribution_type"][0]
    channel_sd = channel_params["channel_sd"][0]
    channel_mean = channel_params["channel_mean"][0]
    #num_EPSCs = 1000
    lotbl = -3000 * np.log(np.arange(1, 10001) * 0.0001)
    count = 1
    iCI = .56
    if len(current_params) != 0:
        iCP = iCI * current_params["iCP_Multiplier"][0]
    else:
        iCP = 0
    # iSC = .56 #Calculated using the known conductance and holding potential of -70mV
    all_EPSCs = []
    all_total_channel_nums = []
    all_open_channel_nums = []
    CP_channels = []
    CI_channels = []
    cp_channel_nums = []
    ci_channels_nums = []
    #glutamate_distribution_type = glutamate_params["distribution_type"]
    glutamate_tracker = []
    peak_CP_channels_open = []
    peak_CI_channels_open = []
    with tqdm(total=num_EPSCs, desc=f"Processing EPSCs for CH:{channel_distribution}  glut:{glut_distribution} ") as pbar:

        while count < num_EPSCs: #marking EPSCs
            if glutamate_params["distribution_type"][0] == "fixed_value":
                if glutamate_params["fixed_value"][0] !=None:
                    glutamate_scale = glutamate_params["fixed_value"][0]
                else:
                    print("Glutamate fixed value not set!")
            elif glutamate_params["distribution_type"][0] == "normal":
                    glutamate_scale = pick_glutamate_scale(glutamate_params)


            glutamate_tracker.append(glutamate_scale)
            AgPulse = Agonist_Pulse(glutamate_scale,steady=agonist_steady,second_pulse=double_agonist)
            indx = 1
            # AgScale = random.uniform(1, 2)
            # AgPulseScaled = AgScale * AgPulse
            all_Channels_Arrays = []  # List to store all channels for a single EPSC
            if channel_params["distribution_type"][0] == "fixed_value":
                num_channels = channel_params["fixed_value"][0]
            else:
                num_channels = pick_number_channels(channel_distribution, sd=channel_sd, mean=channel_mean)
            #num_channels = pick_number_channels(method=channel_distribution, mog_params=mog_params)
            time = np.linspace(0, 16, 800)
            # if "diffusion" in glutamate_params:
            #     if glutamate_params["diffusion"][0] == True:
            #         # scale_factors = [1,.75,.5]
            #         # step_size = num_channels // len(scale_factors)
            #         # channel_splits = np.arange(step_size, step_size*len(scale_factors), step_size)
            #         if num_channels < 800:
            #             dif_idx = 0
            #         if num_channels >= 800 and num_channels <2200:
            #             dif_idx = 1
            #         if num_channels >= 2200:
            #             dif_idx = 2
            #
            #         scale_factors = [5,3,1]
            #         channel_splits = [100000000000000]
            #
            #
            #     else:
            #         dif_idx = 0
            #         channel_splits = [100000000000000]
            #         scale_factors = [1]
            # else:
            #     dif_idx = 0
            #     channel_splits = [100000000000000]
            #     scale_factors = [1]
            # dif_idx = 0
            while indx < num_channels: #marking  number of channels
                # if indx in channel_splits:
                #     dif_idx +=1
                # AgPulse = scale_factors[dif_idx] * AgPulse
                # print(channel_params.columns)
                if len(current_params) > 0:
                    num_CP = int(num_channels * current_params["CP_Ratio"][0])
                else:
                    num_CP = 0
                # print(f"We have {num_channels} channels and {num_CP} are CP")
                if indx < num_CP:
                    iSC = iCP
                    channel_type = 'CP'
                else:
                    iSC = iCI
                    channel_type = 'CI'
                single_channel_array = np.zeros((800,))
                npl = 0
                next_state = 'StateC0'
                while npl < 800: #800 refers to the number of timepoints
                    if next_state == 'StateC0':
                        npl,next_state = StateC0(npl,AgPulse)

                    if next_state == 'StateC1':
                        npl,next_state = StateC1(npl,lotbl,AgPulse)

                    if next_state == 'StateC2':
                        npl,next_state = StateC2(npl,lotbl)

                    if next_state == 'StateC3':
                        npl,next_state = StateC3(npl,lotbl,AgPulse)

                    if next_state == 'StateC4':
                        npl,next_state = StateC4(npl,lotbl)

                    if next_state == 'StateC5':
                        npl,next_state = StateC5(npl,lotbl)

                    if next_state == 'Open':
                        start_open = npl
                        npl,next_state = Open(npl,lotbl)
                        for i in range (start_open, npl):
                            if i < 800:
                                single_channel_array[i] = iSC
                    if next_state == 'End':
                        break
                indx += 1
                all_Channels_Arrays.append(single_channel_array)
                if channel_type == 'CP':
                    CP_channels.append(single_channel_array)
                else:
                    CI_channels.append(single_channel_array)



            count +=1
            pbar.update(1)
            single_EPSC = np.sum(all_Channels_Arrays, axis=0)
            # peak_channels_open = np.max(single_EPSC) / .56
            CP_portion_EPSC = np.sum(CP_channels,axis=0)
            peak_CP_channels_open.append(np.max(CP_portion_EPSC) / iCP)

            CI_portion_EPSC = np.sum(CI_channels,axis=0)
            peak_CI_channels_open.append( np.max(CI_portion_EPSC) / iCI)

            all_total_channel_nums.append(num_channels)

            # all_open_channel_nums.append(peak_channels_open)

            cp_channel_nums.append(num_CP)
            ci_channels_nums.append(num_channels - num_CP)

            all_EPSCs.append(single_EPSC)
    time = np.linspace(0, 16, 2000)
    # plt.plot(time,AgPulse/10)
    plt.xlabel("Time (ms)")
    plt.ylabel("Current (pA)")
    plt.savefig(f"{channel_distribution}_channels_{glut_distribution}_glut_rawEPSCs.png")

    EPSCs_df = pd.DataFrame(all_EPSCs)


    channel_data_df = pd.DataFrame({
        "Total Channels":all_total_channel_nums,
        "Total CP Channels":cp_channel_nums,
        "Total CI Channels":ci_channels_nums,
        "Open CP Channels":peak_CP_channels_open,
        "Open CI Channels":peak_CI_channels_open,
        "Glutamate Concentration":glutamate_tracker
    })
    if output_file_path == None:
        print("No file path specified. Using the last known...")
        output_file_path = f'C:/Users/jawad/Downloads/Python-EPSC-NSFA-Pipeline/Scripts/Experiments/EPSCs_unspecified.xlsx'
    with pd.ExcelWriter(output_file_path,engine='xlsxwriter') as writer:
        EPSCs_df.T.to_excel(writer, sheet_name="EPSCs", index=False, header=False)
        channel_data_df.to_excel(writer, sheet_name="Channel Data", index=False)
    return EPSCs_df,all_total_channel_nums,channel_data_df

#
#
# EPSC_Calc("uniform",1)
# EPSC_Calc("uniform",10)
# EPSC_Calc("uniform",100)
# EPSC_Calc("normal",1)
# EPSC_Calc("normal",10)
# EPSC_Calc("normal",100)
# EPSC_Calc("normal",channel_sd=560,channel_mean=1800)

# glu_array = [.1,1,10,100,1000]
# n_array = [200, 400, 800, 1600,3200]
#
#
#
# for n in n_array:
#     for conc_glu in glu_array:
#         glutamate_params = pd.DataFrame([{"gl_mean": 3.5, "gl_sd": .5, "distribution_type": "fixed_value","fixed_value":conc_glu}])
#
#         channel_params = pd.DataFrame([{"distribution_type":"fixed_value", "channel_sd":None,"channel_mean":None,"fixed_value":n}])
#
#         EPSC_Calc(num_EPSCs=1000,channel_params=channel_params,glutamate_params=glutamate_params)



