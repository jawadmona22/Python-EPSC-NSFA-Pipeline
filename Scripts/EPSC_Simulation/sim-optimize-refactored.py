from scipy.optimize import minimize
from Scripts.NSFA_Tools.EPSC_Graphic_Utilities import multiple_cdf_generator
import Scripts.NSFA_Tools.EPSC_Graphic_Utilities  as egg
from EPSC_Simulation import EPSC_Calc
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
import optuna
from scipy.stats import ks_2samp


'''

A script to find the optimal simulation parameters for ensuring the CDFs will fit
across physiological and CDF data, refactored

Parameters:
    Channels:
        Distribution Type (Lognormal, Normal, Uniform)
        For Lognormal/Norma: Distribution MU, SD

    Glutamate:
        Distribution Type (Lognormal, Normal, Uniform)
        For Lognormal/Norma: Distribution MU, SD

'''



def generate_meanECDF_data():
    """Generate a dataset that follows the given CDF."""
    ###STEP 1: Generate Mean CDF Data ####
    # ##Many CDF generation
    root = 'C:/Users/jawad/Downloads/Python-EPSC-NSFA-Pipeline/'
    file_name = root + 'data-files/Control_Experimental_EPSCs_Juan.xlsx'
    exp_rise_times = []
    exp_amplitudes = []
    exp_taus = []
    sheet_names = []
    folder_name = 'simulation_optimization/'
    #####Experimental values generation#####
    for sheet in tqdm(range(0,10)):
        EPSCs = pd.read_excel(file_name, sheet_name=sheet)
        excel_file = pd.ExcelFile(file_name)
        all_sheet_names = excel_file.sheet_names
        sheet_name = all_sheet_names[sheet]
        sheet_names.append(sheet_name)

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        max_amplitudes = egg.amplitude_histogram_creator(EPSCs, folder_name, False)
        rise_times = egg.rise_times_histogram_creator(EPSCs, folder_name, False)
        taus_array = egg.tau_graph_generator(EPSCs, folder_name, False,time=6)

        exp_rise_times.append(rise_times)
        exp_amplitudes.append(max_amplitudes)
        exp_taus.append(taus_array)

    #These are the target_x and cdf for the mean CDF of a given type. target_x is a list of x values, target_cdf is a list of probability values for those xs
    rise_target_x, rise_target_cdf = multiple_cdf_generator(exp_rise_times,labels=sheet_names,folder_name=folder_name,plt_show=False, plt_type="Experimental ",data_type="Rise Times",unit="(ms)")
    amp_target_x, amp_target_cdf = multiple_cdf_generator(exp_amplitudes,labels=sheet_names,folder_name=folder_name,plt_show=False, plt_type="Experimental ",data_type="Peak Amplitude",unit="(pA)")
    tau_target_x, tau_target_cdf = multiple_cdf_generator(exp_taus,labels=sheet_names,folder_name=folder_name,plt_show=False, plt_type="Experimental ",data_type="Decay Tau",unit="")


    max_length = max(len(rise_target_x), len(amp_target_x), len(tau_target_x), len(rise_target_cdf),
                     len(amp_target_cdf), len(tau_target_cdf))
    print(max_length)
    # Pad with np.concatenate
    rise_target_x = np.concatenate([rise_target_x, np.full((max_length - len(rise_target_x),), np.nan)])
    amp_target_x = np.concatenate([amp_target_x, np.full((max_length - len(amp_target_x),), np.nan)])
    tau_target_x = np.concatenate([tau_target_x, np.full((max_length - len(tau_target_x),), np.nan)])
    rise_target_cdf = np.concatenate([rise_target_cdf, np.full((max_length - len(rise_target_cdf),), np.nan)])
    amp_target_cdf = np.concatenate([amp_target_cdf, np.full((max_length - len(amp_target_cdf),), np.nan)])
    tau_target_cdf = np.concatenate([tau_target_cdf, np.full((max_length - len(tau_target_cdf),), np.nan)])

    cdf_df = pd.DataFrame({
        'rise_x':rise_target_x ,
        'amp_x': amp_target_x,
        'tau_x': tau_target_x,
        'rise_cdf':rise_target_cdf,
        'amp_cdf': amp_target_cdf,
        'tau_cdf':tau_target_cdf

    })

    print(cdf_df)

    cdf_df.to_pickle('mean_cdf_6_30.pkl')

    return cdf_df


def empirical_cdf(data):
    """Calculate the empirical CDF from raw data."""
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, cdf


def two_sample_ks_test(mean_cdf_x, mean_cdf_values, simulation_data):
    # Step 1: Compute the empirical CDF of the simulation data
    sim_sorted, sim_cdf = empirical_cdf(simulation_data)

    # Step 2: Use the x-values from the mean CDF, or create a common x-range
    common_x = np.union1d(mean_cdf_x, sim_sorted)  # Common x-range from both datasets

    # Step 3: Interpolate both CDFs on the common x-range
    mean_cdf_interp = np.interp(common_x, mean_cdf_x, mean_cdf_values)
    sim_cdf_interp = np.interp(common_x, sim_sorted, sim_cdf)
    # print(f"SIM: {sim_cdf_interp}")
    # print(f"MEAN: {mean_cdf_interp}")

    # Handle NaN values in the interpolated CDFs (just in case)
    mean_cdf_interp = np.nan_to_num(mean_cdf_interp, nan=1)
    sim_cdf_interp = np.nan_to_num(sim_cdf_interp, nan=1)

    # Step 4: Compute the KS statistic as the maximum absolute difference
    ks_statistic = np.max(np.abs(mean_cdf_interp - sim_cdf_interp))

    return ks_statistic, common_x, mean_cdf_interp, sim_cdf_interp


def objective(trial):
    mean_channels = trial.suggest_int('mean_channels', 2000, 2200)
    std_channels = trial.suggest_int('std_channels', 700, 750)
    # mean_glutamate = trial.suggest_int('mean_glutamate',2, 3)
    # std_glutamate = trial.suggest_float('std_glutamate', 1, 3)


    channel_params = pd.DataFrame(
        [{"distribution_type": "normal", "channel_sd": 700, "channel_mean": 2200, "fixed_value": None}])

    glutamate_params = pd.DataFrame(
        [{"gl_mean": 2.5, "gl_sd": 1, "distribution_type": "normal", "fixed_value": None,"continuous":True}])

    simulation_EPSCs, num_channels, _ = EPSC_Calc(
        num_EPSCs=1000,
        glutamate_params=glutamate_params,
        channel_params=channel_params, folder_path='simulation_optimization/'
    )
    simulation_EPSCs = simulation_EPSCs.T
    # simulation_EPSCs = pd.read_excel('EPSCs_unspecified.xlsx')
    simulation_rise_times = egg.rise_times_histogram_creator(simulation_EPSCs, plt_show=False,folder_name='simulation_optimization')
    simulation_amplitudes = egg.amplitude_histogram_creator(simulation_EPSCs, plt_show=False,folder_name='simulation_optimization')



    # Compute the KS statistic between simulated and real data
    rise_ks_stat, rise_common_x, rise_cdf1_interp, rise_cdf2_interp = two_sample_ks_test(mean_rise_x, mean_rise_cdf_values, simulation_rise_times)
    amp_ks_stat, amp_common_x, amp_cdf1_interp, amp_cdf2_interp = two_sample_ks_test(mean_amp_x, mean_amp_cdf_values,
                                                                                     simulation_amplitudes)
    print(f"Trial {trial.number} | KS_rise: {rise_ks_stat:.4f}, KS_amp: {amp_ks_stat:.4f}")



    #Plots for debugging
    # Amp CDF Comparison
    plt.close("all")
    plt.figure(figsize=(8, 6))
    plt.plot(amp_common_x, amp_cdf1_interp, label="Mean Amplitude CDF", color='b')
    plt.plot(amp_common_x, amp_cdf2_interp, label="Simulation Amplitude CDF", color='r', linestyle='--')
    plt.xlabel('Amplitude (pA)')
    plt.ylabel('CDF')
    plt.title('Comparison of Mean CDF and Simulation CDF - Amplitude')
    plt.legend()
    plt.grid(False)
    plt.show()
    # Rise CDF Comparison
    plt.close("all")
    plt.figure(figsize=(8, 6))
    plt.plot(rise_common_x, rise_cdf1_interp, label="Mean Rise Time CDF", color='b')
    plt.plot(rise_common_x, rise_cdf2_interp, label="Simulation Rise Time CDF", color='r', linestyle='--')
    plt.xlabel('Rise Time (ms)')
    plt.ylabel('CDF')
    plt.title('Comparison of Mean CDF and Simulation CDF - Rise Time')
    plt.legend()
    plt.grid(False)
    plt.show()
    return rise_ks_stat, amp_ks_stat # lower is better





if __name__ == "__main__":
    #--- First, generating the CDF to check against ---#
    # cdf_df = generate_meanECDF_data()


    mean_cdf = pd.read_pickle("mean_cdf_6_30.pkl")
    print(mean_cdf.head())  # param_x indicates the x-axis value of the parameter, param_cdf is the percentage value


    mean_rise_x = mean_cdf['rise_x']
    mean_rise_cdf_values = mean_cdf['rise_cdf']
    mean_amp_x = mean_cdf['amp_x']
    mean_amp_cdf_values = mean_cdf['amp_cdf']
    mean_tau_x = mean_cdf['tau_x']
    mean_tau_cdf_values = mean_cdf['tau_cdf']

    #------ Setting up the OPTUNA study --- #
    study = optuna.create_study(direction="minimize")
    study = optuna.create_study(
        directions=["minimize", "minimize"],  # Multi-objective
    )
    study.optimize(objective, n_trials=100)  # Increase n_trials for better results

    print("Best Parameters:", study.best_trials)
    # print("Best KS Statistic:", study.best_values)

    # ---- Visualizing the Pareto Front ---- #
    trials = [t for t in study.trials if t.values is not None]

    ks_rise_vals = [t.values[0] for t in trials]
    ks_amp_vals = [t.values[1] for t in trials]

    plt.figure(figsize=(8, 6))
    plt.scatter(ks_rise_vals, ks_amp_vals, c='blue', alpha=0.6)
    plt.xlabel("KS Rise Time")
    plt.ylabel("KS Amplitude")
    plt.title("Pareto Frontier: Rise Time vs Amplitude")
    plt.show()

    # Access the Pareto front
    front = study.best_trials

    for trial in front:
        print(f"KS Rise: {trial.values[0]:.4f}, KS Amp: {trial.values[1]:.4f}, Params: {trial.params}")







