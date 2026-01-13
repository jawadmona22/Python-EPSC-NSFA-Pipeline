from scipy.optimize import minimize
from Scripts.NSFA_Tools.EPSC_Graphic_Utilities import multiple_cdf_generator
import Scripts.NSFA_Tools.EPSC_Graphic_Utilities  as egg
from EPSC_Simulation import EPSC_Calc
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import least_squares


opt_iteration = 0

def generate_meanECDF_data(data_type = "rise_times"):
    """Generate a dataset that follows the given CDF."""
    ###STEP 1: Generate Mean CDF Data ####
    # ##Many CDF generation
    file_name = 'data-files/Control_Experimental_EPSCs_Juan.xlsx'
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
        taus_array = egg.tau_graph_generator(EPSCs, folder_name, False)

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

    return cdf_df
    # # Create interpolation function for the inverse CDF
    # inverse_cdf = interp1d(target_cdf, target_x, kind='linear',
    #                        bounds_error=False,
    #                        fill_value=(target_x[0], target_x[-1]))
    #
    # # Generate uniform random numbers
    # num_samples = 1000
    # uniform_samples = np.random.uniform(0, 1, num_samples)
    #
    # # Transform to match the original distribution
    # synthetic_samples = inverse_cdf(uniform_samples)
    # # Sort the synthetic samples
    # synthetic_samples_sorted = np.sort(synthetic_samples)
    #
    # # Create synthetic CDF using the same x points as target
    # synthetic_cdf = np.zeros_like(target_x)
    # for i, x in enumerate(target_x):
    #     synthetic_cdf[i] = np.sum(synthetic_samples_sorted <= x) / len(synthetic_samples_sorted)
    #
    # # Plot to compare distributions
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    #
    # # Plot histogram of synthetic data
    # ax1.hist(synthetic_samples, bins=50, density=True, alpha=0.7, label='Synthetic')
    # ax1.set_title('Histogram of Synthetic Data')
    # ax1.legend()
    #
    # # Plot CDFs
    # ax2.plot(target_x, target_cdf, label='Original CDF', alpha=0.7)
    # ax2.plot(target_x, synthetic_cdf, label='Synthetic CDF', alpha=0.7)
    # ax2.set_title('CDF Comparison')
    # ax2.legend()
    #
    # plt.tight_layout()
    # plt.savefig("simulation_optimization/Synthetic_CDF_Plots.png")
    # plt.show()
    #
    # # Calculate KS statistic to measure the difference
    # ks_stat = np.max(np.abs(target_cdf - synthetic_cdf))
    # print(f"KS statistic between original and synthetic: {ks_stat:.4f}")

    # return synthetic_samples, target_x, target_cdf, synthetic_cdf




def generate_synthetic_data( num_samples=1000):
    """
    Generate synthetic data that follows the same distribution as the original data.

    Parameters:
    original_data (array-like): Original experimental data
    num_samples (int): Number of synthetic samples to generate

    Returns:
    array: Generated synthetic samples
    """
    # Create empirical CDF
    x_cdf, y_cdf = generate_meanECDF_data(data_type="rise_times")

    # Create interpolation function for the inverse CDF
    inverse_cdf = interp1d(y_cdf, x_cdf, kind='linear',
                           bounds_error=False,
                           fill_value=(x_cdf[0], x_cdf[-1]))

    # Generate uniform random numbers
    uniform_samples = np.random.uniform(0, 1, num_samples)

    # Transform to match the original distribution
    synthetic_samples = inverse_cdf(uniform_samples)

    return synthetic_samples

def ks_p_value(D, n, m):
    """
    Approximate the p-value for the KS statistic based on sample sizes n and m.
    """
    # Approximation of the distribution of D for large sample sizes (asymptotic distribution)
    print(f"n: {n} m: {m}")

    return np.exp(-2 * (D ** 2) * (n * m) / (n + m))
def optimize_epsc_params(mean_cdf, initial_guess, bounds):


    def objective(params, mean_cdf):
        channel_sd, channel_mean,gl_sd, gl_mean = params
        print(params)
        print(f"Trying parameters: channel_sd={channel_sd:.3f}, channel_mean={channel_mean:.3f}, [gl_sd] = {gl_sd: .3f}, gl_mean={gl_mean:.3f}")
        channel_params = pd.DataFrame(
            [{"distribution_type": "normal", "channel_sd": 560, "channel_mean": 1800, "fixed_value": None}])

        glutamate_params = pd.DataFrame(
            [{"gl_mean": 3.5, "gl_sd": .5, "distribution_type": "lognormal", "fixed_value": None}])
        # simulation_EPSCs, num_channels, _ = EPSC_Calc(
        #     num_EPSCs=1000,
        #     glutamate_params=glutamate_params,
        #     channel_params=channel_params
        # )
        simulation_EPSCs = pd.read_excel('C:/Users/jawad/Downloads/Python-EPSC-NSFA-Pipeline/Scripts/Experiments/Simulation_Glu_N_Settings_Investigation/EPSC_Data/EPSCS_CDFfit_1000_sd1.xlsx',sheet_name=0,index_col=None)
        # simulation_EPSCs = simulation_EPSCs.T #transpose into correct shape (rows = time)

        folder_name = 'C:/Users/jawad/Downloads/Python-EPSC-NSFA-Pipeline/Scripts/Experiments/Simulation_Glu_N_Settings_Investigation'
        simulation_rise_times = egg.rise_times_histogram_creator(simulation_EPSCs,folder_name=folder_name,plt_show=False)

        simulation_amplitudes = egg.amplitude_histogram_creator(simulation_EPSCs,folder_name=folder_name,plt_show=False)
        simulation_taus = egg.tau_graph_generator(simulation_EPSCs,folder_name=folder_name,plt_show=False)

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
            #print(f"SIM: {sim_cdf_interp}")
            #print(f"MEAN: {mean_cdf_interp}")

            # Handle NaN values in the interpolated CDFs (just in case)
            mean_cdf_interp = np.nan_to_num(mean_cdf_interp, nan=1)
            sim_cdf_interp = np.nan_to_num(sim_cdf_interp, nan=1)

            # Step 4: Compute the KS statistic as the maximum absolute difference
            ks_statistic = np.max(np.abs(mean_cdf_interp - sim_cdf_interp))
            gap_difference = np.argmax(np.abs(mean_cdf_interp - sim_cdf_interp))
            print(f"Index of largest difference: { gap_difference}")

            return ks_statistic, common_x, mean_cdf_interp, sim_cdf_interp,gap_difference





        # Step 1: Extract the mean CDF
        mean_rise_x = mean_cdf['rise_x']
        mean_rise_cdf_values = mean_cdf['rise_cdf']
        mean_amp_x = mean_cdf['amp_x']
        mean_amp_cdf_values = mean_cdf['amp_cdf']
        mean_tau_x = mean_cdf['tau_x']
        mean_tau_cdf_values = mean_cdf['tau_cdf']

        # Step 2: Run the two-sample KS test with raw simulation data and the mean CDF
        rise_ks_stat, rise_common_x, rise_cdf1_interp, rise_cdf2_interp, gap_difference = two_sample_ks_test(mean_rise_x, mean_rise_cdf_values,
                                                                         simulation_rise_times)
        amp_ks_stat, amp_common_x, amp_cdf1_interp, amp_cdf2_interp, _ = two_sample_ks_test(mean_amp_x, mean_amp_cdf_values,
                                                                         simulation_amplitudes)
        tau_ks_stat, tau_common_x, tau_cdf1_interp, tau_cdf2_interp, _ = two_sample_ks_test(mean_tau_x, mean_tau_cdf_values,
                                                                         simulation_taus)

        # Print the KS statistic
        print(f"Rise KS Statistic: {rise_ks_stat}")
        print(f"Amp KS Statistic: {amp_ks_stat}")
        print(f"Tau KS Statistic: {tau_ks_stat}")


        #Amp CDF Comparison
        plt.close("all")
        plt.figure(figsize=(8, 6))
        plt.plot(amp_common_x, amp_cdf1_interp, label="Mean Amplitude CDF", color='b')
        plt.plot(amp_common_x, amp_cdf2_interp, label="Simulation Amplitude CDF", color='r', linestyle='--')
        plt.xlabel('Amplitude (pA)')
        plt.ylabel('CDF')
        plt.title('Comparison of Mean CDF and Simulation CDF - Amplitude')
        plt.legend()
        plt.grid(False)
        plt.savefig('C:/Users/jawad/Downloads/Python-EPSC-NSFA-Pipeline/Scripts/Experiments/RiseTime_Pooling_Investigation/amp_cdf_comparison.png')
        plt.show()

        #Rise time CDF comparison
        plt.figure(figsize=(8, 6))
        plt.plot(rise_common_x, rise_cdf1_interp, label="Mean Rise CDF", color='b')
        plt.plot(rise_common_x, rise_cdf2_interp, label="Simulation Rise CDF", color='r', linestyle='--')
        plt.vlines(rise_common_x[gap_difference],ymin=0,ymax=1,linestyles='--')
        plt.xlabel('Time (ms)')
        plt.ylabel('CDF')
        plt.title('Comparison of Mean CDF and Simulation CDF - Rise Times')
        plt.legend()
        plt.grid(False)
        plt.savefig('C:/Users/jawad/Downloads/Python-EPSC-NSFA-Pipeline/Scripts/Experiments/RiseTime_Pooling_Investigation/rise_time_cdf_comparison.png')
        plt.show()

        #Tau CDF Comparison
        plt.figure(figsize=(8, 6))
        plt.plot(tau_common_x, tau_cdf1_interp, label="Mean Tau CDF", color='b')
        plt.plot(tau_common_x, tau_cdf2_interp, label="Simulation Tau CDF", color='r', linestyle='--')
        plt.xlabel('Time (ms)')
        plt.ylabel('CDF')
        plt.title('Comparison of Mean CDF and Simulation CDF - Tau')
        plt.legend()
        plt.grid(False)
        plt.savefig('C:/Users/jawad/Downloads/Python-EPSC-NSFA-Pipeline/Scripts/Experiments/RiseTime_Pooling_Investigation/tau_cdf_comparison.png')
        plt.show()
        global opt_iteration
        opt_iteration +=1
        # if opt_iteration % 10 == 0:
        #     plt.show()

        #return (2 * amp_ks_stat) + rise_ks_stat

        return amp_ks_stat


    # Optimization options
    options = {
        'maxiter': 100,
        'disp': True,  # Show optimization progress
        'ftol': 1e-6  # Function tolerance for convergence
    }

    # # Run optimization
    result = minimize(
        objective,
        initial_guess,
        args=(mean_cdf,),
        method='L-BFGS-B',  # Works well with bounds
        bounds=bounds,
        options=options
    )
    # print(bounds)
    # result = least_squares(
    #     objective,
    #     initial_guess,
    #     args=(mean_cdf,),
    #     method='trf',
    #     bounds=bounds,
    #     verbose=2
    # )
    if result.success:
        print(f"Optimization successful!")
        print(f"Found optimal parameters: SD={result.x[0]:.3f}, Mean={result.x[1]:.3f}")
        print(f"Final KS statistic: {result.fun:.6f}")
    else:
        print("Optimization did not converge!")
        print(f"Best parameters found: SD={result.x[0]:.3f}, Mean={result.x[1]:.3f}")
        print(f"Final KS statistic: {result.fun:.6f}")
        print(f"Message: {result.message}")

    return result.x, result






if __name__ == "__main__":
    # mean_cdf = generate_meanECDF_data(data_type="rise_times")
    # #Run once to save
    # synth_df = pd.DataFrame(synthetic_samples)
    # mean_cdf_df.to_pickle("mean_cdf_data.pkl")
    mean_cdf = pd.read_pickle("mean_cdf_6_30.pkl")
    bounds = [(100,1000),(500,4000),(.5,10),(1,15)]
    initial_guess = [560,1800,.5,3.5]  # Channel SD, channel Mean, and glutamate SD, glutamate scale Mean

    # Run optimization
    optimal_params, result = optimize_epsc_params(
        mean_cdf,
        initial_guess=initial_guess,
        bounds=bounds
    )
    # glutamate_params = pd.DataFrame([{"gl_mean": 3.5, "gl_sd": .5, "distribution_type": "lognormal"}])
    #
    # simulation_EPSCs, num_channels = EPSC_Calc(
    #     channel_distribution="uniform",
    #     glutamate_params=glutamate_params
    # )
    # simulation_EPSCs = simulation_EPSCs.T  # transpose into correct shape (rows = time)
    # simulation_EPSCs.to_pickle("EPSCs-uniformchannels.pkl")


