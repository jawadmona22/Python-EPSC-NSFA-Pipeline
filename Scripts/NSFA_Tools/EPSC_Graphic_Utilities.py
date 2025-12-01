import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from scipy.stats import ks_2samp
import glob
import seaborn as sns
# Define the decaying exponential function
def exp_decay(t, A, tau, C):
    return A * np.exp(-t / tau) + C
def tau_graph_generator(data,folder_name, plt_show=False,debug=False,time=6): #Time in ms
    taus = []
    print(f"Using time {time} ms")
    for col in data.columns:
        t = np.linspace(0, time, data.shape[0])
        y = data[col].values

        # Find the index of the peak
        peak_index = np.argmax(y)
        # print(f"Time of Peak: {peak_index}")
        # Use only the values after the peak for curve fitting
        t_post_peak = t[peak_index+2:]
        y_post_peak = y[peak_index+2:]

        try:
            popt, _ = curve_fit(exp_decay, t_post_peak, y_post_peak, p0=(1, 1, 1))  # Initial guesses for A, tau, C
            A_fit, tau_fit, C_fit = popt
            # print(tau_fit)
            if debug:
                #Check function fit
                plt.figure(figsize=(8,6))
                plt.scatter(t_post_peak,y_post_peak) #Plot the data
                x_fit = np.linspace(min(t_post_peak),max(t_post_peak),100)
                y_fit  = exp_decay(x_fit,A_fit,tau_fit,C_fit)
                plt.xlabel("Time (ms)")
                plt.ylabel("Current (pA)")
                plt.plot(x_fit,y_fit,label=f"Fitted Curve for tau: {tau_fit}")
                plt.show()

            if tau_fit < 0:
                taus.append(0)
            else:
                taus.append(tau_fit)
        except RuntimeError:
            taus.append(0)  # Append 0 if the fit fails



    if plt_show:
        plt.figure()
        plt.hist(taus, bins=10, edgecolor='black')
        plt.title('Tau Histogram')
        plt.xlabel("Decay Tau (ms)")
        plt.ylabel("Frequency")
        plt.savefig(f"{folder_name}/tau_histogram.png")
        plt.show()

    return taus



def amplitude_histogram_creator(df,folder_name,plt_show = False):
    max_amplitudes = df.abs().max(axis=0)
    max_amplitudes.to_excel('max_amplitudes.xlsx',index_label='Trace', header=['Max Amplitude'])

    if plt_show == True:
        plt.figure()
        plt.hist(max_amplitudes, bins=10, edgecolor='black')
        plt.xlabel('EPSC Amplitudes (pA)')
        plt.ylabel('Frequency')
        plt.title('EPSC Amplitude Bins')
        plt.savefig(f'{folder_name}/peak_amplitude_histograms.png', dpi=300, bbox_inches='tight')
        plt.show()
    return max_amplitudes

def rise_times_histogram_creator(df,folder_name,plt_show = False,sampling_rate = .02):
    max_amplitudes = df.abs().max(axis=0)
    rise_times = []
    for col in df.columns:
        trace = df[col].abs()
        max_amp = trace.max()
        threshold_10 = 0.1 * max_amp
        threshold_90 = 0.9 * max_amp

        # Find where the signal first crosses the 10% threshold
        above_10 = trace >= threshold_10
        above_90 = trace >= threshold_90

        try:
            idx_10 = above_10.idxmax()  # First index where >= threshold_10
            idx_90 = above_90.idxmax()

            if idx_10 == 0:
                time = idx_90 * sampling_rate
                rise_times.append(time)
                continue  # Can't interpolate at the start of trace

            # Interpolate time_10
            prev_idx_10 = idx_10 - 1
            x0, y0 = prev_idx_10, trace[prev_idx_10]
            x1, y1 = idx_10, trace[idx_10]
            time_10_interp = x0 + (threshold_10 - y0) / (y1 - y0)

            # Interpolate time_90
            prev_idx_90 = idx_90 - 1
            x0, y0 = prev_idx_90, trace[prev_idx_90]
            x1, y1 = idx_90, trace[idx_90]
            time_90_interp = x0 + (threshold_90 - y0) / (y1 - y0)

            # Convert to seconds (assuming 0.02 ms per sample)
            rise_time = (time_90_interp - time_10_interp) * 0.02
            rise_times.append(rise_time)

        except Exception as e:
            print(f"Interpolation failed for column {col}: {e}")

    print(f"Rise Time shape: {len(rise_times)}")
    print(f"Trace shape: {df.shape}")

    rise_times_df = pd.DataFrame({
        'Trace':df.columns,
        'Rise Time (10% to 90%)': rise_times
    })

    rise_times_df.to_excel(f'rise_times.xlsx',index=False)
    if plt_show == True:
        plt.figure()
        plt.hist(rise_times, bins=10, edgecolor='black')
        plt.xlabel('Rise Times (ms)')
        plt.title(f'Rise Times Histogram')
        plt.ylabel('Frequency')
        plt.show()
        plt.savefig(f'rise_times_histogram.png')
    return rise_times

def rise_time_amplitude_scatterplot(rise_time,max_amplitude, folder_name,plt_show=False):
    plt.figure()
    plt.scatter(rise_time,max_amplitude)
    plt.xlabel("Rise Time 10%-90% of Peak (ms)")
    plt.ylabel("EPSC Peak Amplitude (pA)")
    plt.title("Simulated EPCSs Amplitude vs. Rise Time")
    plt.savefig(f"{folder_name}/rise_amplitude_scatter.png")
    if plt_show:
        plt.show()


def cdf_generator(data,folder_name,plt_show=False):
    sorted_data = np.sort(data)
    cdf = np.searchsorted(sorted_data, sorted_data, side="right") / len(data)

    if plt_show:
        plt.figure()
        plt.step(sorted_data, cdf, where='post', color='green', label='Empirical CDF')
        plt.title("Empirical Cumulative Density Function for Rise Times")
        plt.xlabel("Rise Times (ms)")
        plt.savefig(f'{folder_name}/rise_times_CDF.png')
        plt.show()
    return cdf



def generate_meanECDF_data(file_name,folder_name, use_mean=True,use_median=False,plt_show=False,time=6):  #This file should be a .xlsx spreadsheet with tabs labeled with the names of the cells
    if use_median:
        print("Using median")
    """Generate a dataset that follows the given CDF."""
    ###STEP 1: Generate Mean CDF Data ####
    # ##Many CDF generation
    exp_rise_times = []
    exp_amplitudes = []
    exp_taus = []
    sheet_names = []
    #####Experimental values generation#####
    for sheet in tqdm(range(0,10)):
        EPSCs = pd.read_excel(file_name, sheet_name=sheet)
        excel_file = pd.ExcelFile(file_name)
        all_sheet_names = excel_file.sheet_names
        sheet_name = all_sheet_names[sheet]
        sheet_names.append(sheet_name)

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        max_amplitudes = amplitude_histogram_creator(EPSCs, folder_name, False)
        rise_times = rise_times_histogram_creator(EPSCs, folder_name, False)
        taus_array = tau_graph_generator(EPSCs, folder_name, False,time=time)

        exp_rise_times.append(rise_times)
        exp_amplitudes.append(max_amplitudes)
        exp_taus.append(taus_array)

    #These are the target_x and cdf for the mean CDF of a given type. target_x is a list of x values, target_cdf is a list of probability values for those xs
    rise_target_x, rise_target_cdf = multiple_cdf_generator(exp_rise_times,labels=sheet_names,folder_name=folder_name,plt_show=plt_show, plt_type="Experimental ",data_type="Rise Times",unit="(ms)",use_mean=use_mean,use_median=use_median)
    amp_target_x, amp_target_cdf = multiple_cdf_generator(exp_amplitudes,labels=sheet_names,folder_name=folder_name,plt_show=plt_show, plt_type="Experimental ",data_type="Peak Amplitude",unit="(pA)",use_mean=use_mean,use_median=use_median)
    tau_target_x, tau_target_cdf = multiple_cdf_generator(exp_taus,labels=sheet_names,folder_name=folder_name,plt_show=plt_show, plt_type="Experimental ",data_type="Decay Tau",unit="",use_mean=use_mean,use_median=use_median)


    max_length = max(len(rise_target_x), len(amp_target_x), len(tau_target_x), len(rise_target_cdf),
                     len(amp_target_cdf), len(tau_target_cdf))
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


def multiple_cdf_generator(datasets,labels,folder_name,plt_show=True,plt_type = "",data_type = "",unit = "",use_mean=True,use_median=False):
    plt.figure()

    all_sorted_data = []
    all_ecdfs = []
    for data, label in zip(datasets,labels):
        sorted_data = np.sort(data) #rise times, max amplitudes, etc.
        #In this cdf, every value represents the fraction of data points <= that value
        cdf = np.searchsorted(sorted_data, sorted_data, side="right") / len(data)
        #cdf = np.arange(1,len(sorted_data)+1)/len(sorted_data)
        plt.step(sorted_data,cdf,where='post',label=label,alpha=.5)
        all_sorted_data.append(sorted_data)
        all_ecdfs.append(cdf)

    #Ensure we get the same length for averaging
    common_x_values = np.unique(np.concatenate(all_sorted_data))

    # Compute the mean ECDF on these common x-values

    median_ecdf = np.zeros_like(common_x_values)
    mean_ecdf = np.zeros_like(common_x_values)
    

    for i, x_value in enumerate(common_x_values):
        # Interpolate each ECDF to the common x-values (sorted data)
        values_at_x = [np.interp(x_value, sorted_data, cdf) for sorted_data, cdf in zip(all_sorted_data, all_ecdfs)]
        mean_ecdf[i] = np.mean(values_at_x)
        median_ecdf[i] = np.median(values_at_x)
    if use_mean:
        plt.plot(common_x_values, mean_ecdf, label="Mean ECDF", color='black', linewidth=3)
    if use_median:
        plt.plot(common_x_values, median_ecdf, label="Median ECDF", color='red', linewidth=3)
        median_df = pd.DataFrame({
            'x_values': common_x_values,
            'median_ecdf': median_ecdf
        })
        median_df.to_excel(f'{folder_name}/median_ecdf.xlsx', index=False)
    plt.title(f"Empirical Cumulative Density Functions - {plt_type + data_type}")
    plt.xlabel(f"{data_type + ' ' + unit}")
    plt.ylabel("ECDF")
    plt.legend()
    plt.savefig(f'{folder_name}/multiple_CDFs-{plt_type + "-" +data_type}.png',bbox_inches='tight')

    if plt_show:
        plt.show()

    return common_x_values,mean_ecdf
def tau_amplitude_scatterplot(taus,max_amplitudes, folder_name,plt_show =False):
    plt.figure()
    plt.scatter(taus,max_amplitudes)
    plt.title("EPSC Amplitudes vs Decay Constants")
    plt.xlabel("Tau (ms)")
    plt.ylabel("EPSC Amplitude (pA)")
    plt.savefig(f"{folder_name}/tau_amplitude_scatterplot.png")
    if plt_show:
        plt.show()

def EPSC_Plot(EPSCs,folder_name,plt_show=False):
    plt.figure()
    time = np.linspace(0, 16, EPSCs.shape[0])
    plt.plot(time,EPSCs)
    print(EPSCs.shape)
    plt.xlabel("Time (ms)")
    plt.title("Simulated EPSCs")
    plt.ylabel("EPSC Amplitude (pA)")
    plt.savefig(f"{folder_name}/EPSCs_plotted.png")
    if plt_show:
        plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

def ks_matrix_generator(experimental, simulation,title=""):
    num_exp = len(experimental)
    num_sim = len(simulation)
    ks_matrix = np.zeros((num_exp, num_sim))
    p_matrix = np.zeros((num_exp, num_sim))

    # Compute KS statistics and p-values for each pair of datasets
    for i, exp_data in enumerate(experimental):
        for j, sim_data in enumerate(simulation):
            ks_statistic, p_value = ks_2samp(exp_data, sim_data)
            ks_matrix[i, j] = ks_statistic  # Store KS statistic
            p_matrix[i, j] = p_value  # Store p-value

    # Create row and column labels
    exp_labels = [f"Exp_{i + 1}" for i in range(num_exp)]
    sim_labels = ["100mM","10mM","1mM"]

    # Convert to Pandas DataFrame with proper labels
    ks_df = pd.DataFrame(ks_matrix, index=exp_labels, columns=sim_labels)
    p_df = pd.DataFrame(p_matrix, index=exp_labels, columns=sim_labels)

    # Plot the KS statistic matrix as a table
    plt.figure(figsize=(8, 6))
    plt.table(cellText=ks_df.values, colLabels=ks_df.columns, rowLabels=ks_df.index,
              cellLoc='center', loc='center', colColours=['#f2f2f2']*ks_df.shape[1])
    plt.title('KS Statistic Matrix')
    plt.axis('off')  # Hide the axes
    plt.savefig(f'KS_Matrix_{title}')
    plt.show()

    # Plot the P-value matrix as a table
    plt.figure(figsize=(8, 6))
    plt.table(cellText=p_df.values, colLabels=p_df.columns, rowLabels=p_df.index,
              cellLoc='center', loc='center', colColours=['#f2f2f2']*p_df.shape[1])
    plt.title('P-Value Matrix')
    plt.savefig(f'KS-PValueMatrix-{title}')

    plt.axis('off')  # Hide the axes
    plt.show()

def compare_IEM_control():
    IEM_excel_file = pd.ExcelFile("IEM_Data/IEM_EPSCs.xlsx")
    IEM_sheet_names = IEM_excel_file.sheet_names
    control_excel_file = pd.ExcelFile("data-files/Many_EPSCs_Juan.xlsx")
    control_sheet_names = control_excel_file.sheet_names
    IEM_mean_rise_times = np.array([])
    control_mean_rise_times = np.array([])
    IEM_mean_amplitudes = np.array([])
    control_mean_amplitudes = np.array([])
    IEM_ordered_names = []
    control_ordered_names = []
    IEM_raw_rise_times = []
    IEM_raw_amplitudes = []
    IEM_raw_taus = []
    control_raw_rise_times = []
    control_raw_amplitudes = []
    control_raw_taus = []
    for i,sheet_name in enumerate(IEM_sheet_names):
        split_name = sheet_name.split("_")[0] #Name of control cell
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

        if split_name in control_sheet_names:
            rise_times = []
            max_amplitudes = []
            decay_taus = []
            labels = []
            print(f"Sheet to compare: control - {split_name} and IEM - {sheet_name} ")
            IEM_data = pd.read_excel("IEM_Data/IEM_EPSCs.xlsx",sheet_name = sheet_name)
            control_data = pd.read_excel("data-files/Many_EPSCs_Juan.xlsx",sheet_name = split_name)
            #IEM data read-in
            IEM_rise_times = rise_times_histogram_creator(df=IEM_data,folder_name = "IEM_Data/")
            IEM_peak_amplitudes = amplitude_histogram_creator(df=IEM_data,folder_name = "IEM_Data/")
            IEM_decay_taus = tau_graph_generator(IEM_data,folder_name = "IEM_Data/")
            #Control data read-in
            control_rise_times = rise_times_histogram_creator(df=control_data,folder_name="IEM_Data/")
            control_peak_amplitudes = amplitude_histogram_creator(df=control_data,folder_name = "IEM_Data/")
            control_decay_taus = tau_graph_generator(control_data,folder_name = "IEM_Data/")
            #CDF generation
            rise_times.append(IEM_rise_times)
            rise_times.append(control_rise_times)

            #Keep for scatter later
            IEM_raw_rise_times.append(IEM_rise_times)
            IEM_raw_amplitudes.append(IEM_peak_amplitudes)
            IEM_raw_taus.append(IEM_decay_taus)
            control_raw_rise_times.append(control_rise_times)
            control_raw_amplitudes.append(control_peak_amplitudes)
            control_raw_taus.append(control_decay_taus)


            ##Plot rise_times/peak amplitude: IEM
            plt.figure(figsize=(6,5))
            plt.scatter(IEM_rise_times,IEM_peak_amplitudes,color=colors[i])
            plt.xlabel("Rise Times (ms)")
            plt.ylabel("Max Amplitude (pa)")
            plt.title(f"{sheet_name} Max Amplitude versus Rise Times")
            plt.savefig(f'scatter_plots_ampvs/{sheet_name}_risevsamp')

            ##Plot decay taus/pead amplitudes
            plt.figure(figsize=(6, 5))
            plt.scatter(IEM_decay_taus, IEM_peak_amplitudes,color=colors[i])
            plt.xlabel("Decay Taus")
            plt.ylabel("Max Amplitude (pa)")
            plt.title(f"{sheet_name} Max Amplitude versus Decay Taus")
            plt.savefig(f'scatter_plots_ampvs/{sheet_name}_tauvsamp')

            ##Plot rise_times/peak amplitude: IEM
            plt.figure(figsize=(6, 5))
            plt.scatter(control_rise_times, control_peak_amplitudes,marker = 's',color=colors[i])
            plt.xlabel("Rise Times (ms)")
            plt.ylabel("Max Amplitude (pa)")
            plt.title(f"{split_name} Max Amplitude versus Rise Times")
            plt.savefig(f'scatter_plots_ampvs/{split_name}_risevsamp')

            ##Plot decay taus/pead amplitudes
            plt.figure(figsize=(6, 5))
            plt.scatter(control_decay_taus, control_peak_amplitudes,marker = 's', color=colors[i])
            plt.xlabel("Decay Taus")
            plt.ylabel("Max Amplitude (pa)")
            plt.title(f"{split_name} Map Amplitude versus Decay Taus")
            plt.savefig(f'scatter_plots_ampvs/{split_name}_tauvsamp')


            # max_amplitudes.append(IEM_peak_amplitudes)
            # max_amplitudes.append(control_peak_amplitudes)
            # decay_taus.append(IEM_decay_taus)
            # decay_taus.append(control_decay_taus)
            # labels.append(sheet_name) #IEM Names
            # labels.append(split_name) #Control Names
            # multiple_cdf_generator(rise_times,labels=labels,folder_name="IEM_Data/IEM-vs-control/",plt_show=False, plt_type=f"IEM vs Control {split_name} ",data_type="Rise Times",unit="(ms)",use_mean=False)
            # multiple_cdf_generator(max_amplitudes,labels=labels,folder_name="IEM_Data/IEM-vs-control/",plt_show=False, plt_type=f"IEM vs Control {split_name} ",data_type="Peak Amplitudes",unit="(pA)",use_mean=False)
            # multiple_cdf_generator(decay_taus,labels=labels,folder_name="IEM_Data/IEM-vs-control/",plt_show=False, plt_type=f"IEM vs Control {split_name} ",data_type="Decay Taus",unit="",use_mean=False)
            #Calculations for mean plotting later
            IEM_mean_rise_times = np.append(IEM_mean_rise_times, np.mean(IEM_rise_times))
            control_mean_rise_times = np.append(control_mean_rise_times, np.mean(control_rise_times))
            IEM_mean_amplitudes = np.append(IEM_mean_amplitudes, np.mean(IEM_peak_amplitudes))
            control_mean_amplitudes = np.append(control_mean_amplitudes, np.mean(control_peak_amplitudes))
            IEM_ordered_names.append(sheet_name)
            control_ordered_names.append(split_name)

    #Delta amplitudes versus rise times plotting
    # colors = plt.cm.viridis(np.linspace(0, 1, len(IEM_ordered_names)))  # Using a colormap
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    percent_delta_amplitudes = np.abs(((IEM_mean_amplitudes - control_mean_amplitudes)/control_mean_amplitudes)) * 100
    percent_delta_rise_times = ((IEM_mean_rise_times- control_mean_rise_times)/control_mean_rise_times)* 100
    plt.close('all')
    plt.figure()
    handles = []
    for x, y, name, color in zip(percent_delta_amplitudes, percent_delta_rise_times, IEM_ordered_names, colors):
        scatter = plt.scatter(x, y, color=color, label=name)
        handles.append(scatter)  # Collect handles for legend

    plt.xlabel("Percent Reduction in Peak Amplitudes (pA)")
    plt.ylabel("Percent Change in Rise Times (ms)")
    plt.title("IEM vs Control")

    # Add a legend with labels
    plt.legend(handles=handles, labels=IEM_ordered_names, title="Cell and Dosage", loc="best")

    plt.savefig("IEM_Data/IEM-vs-control/delta-amplitude.png",bbox_inches='tight')
    plt.show()

    plt.close('all')
    plt.figure()

    for x, y in zip(IEM_raw_rise_times, IEM_raw_amplitudes):
        print(len(x))
        print(len(y))
        scatter = plt.scatter(x, y,color='black')
    plt.xlabel('Rise Times (ms)')
    plt.ylabel('Max Amplitude (pA)')
    plt.title("IEM")
    plt.show()

    plt.close('all')
    for x, y in zip(IEM_raw_taus, IEM_raw_amplitudes):
        print(len(x))
        print(len(y))
        scatter = plt.scatter(x, y,color='black')

    plt.xlabel('Decay Taus')
    plt.ylabel('Max Amplitude (pA)')
    plt.title("IEM")
    plt.show()

    for x, y in zip(control_raw_taus, control_raw_amplitudes):
        print(len(x))
        print(len(y))
        scatter = plt.scatter(x, y, color='black')

    plt.xlabel('Decay Taus')
    plt.ylabel('Max Amplitude (pA)')
    plt.title("Control")

    plt.show()

    for x, y in zip(control_raw_rise_times, control_raw_amplitudes):
        print(len(x))
        print(len(y))
        scatter = plt.scatter(x, y, color='black')

    plt.xlabel('Rise Times (ms)')
    plt.ylabel('Max Amplitude (pA)')
    plt.title("Control")

    plt.show()


    #Pure Amplitudes versus kinetics plotting

    # Function to plot structured scatter plots
    def structured_scatter(data_y, data_x, title, xlabel, ylabel,cell_labels):
        plt.figure(figsize=(6, 5))
        i = 0
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        for x, y in zip(data_x, data_y):
            plt.scatter(x, y, color=colors[i % len(colors)], label=cell_labels[i])
            i+=1
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.show()

    # Individual - IEM, Amplitude vs Decay Taus
    structured_scatter(IEM_raw_amplitudes, IEM_rise_times, "IEM Rise Times vs Max Amplitudes", "Max Amplitude (pA)",
                       "Rise Time (ms)",IEM_ordered_names)
    # Individual - IEM, Amplitude vs Decay Taus
    structured_scatter(IEM_raw_amplitudes, IEM_raw_taus, "IEM Decay Taus vs Max Amplitudes", "Max Amplitude (pA)",
                       "Decay Tau",IEM_ordered_names,IEM_ordered_names)

    # Control - Amplitude vs Rise Time
    structured_scatter(control_raw_amplitudes, control_raw_rise_times, "Control Rise Times vs Max Amplitudes",
                       "Max Amplitude (pA)", "Rise Time (ms)",control_ordered_names)

    # Control - Amplitude vs Decay Tau
    structured_scatter(control_raw_amplitudes, control_raw_taus, "Control Decay Taus vs Max Amplitudes",
                       "Max Amplitude (pA)", "Decay Tau",control_ordered_names)


def main():

    compare_IEM_control()
    # EPSCS = pd.read_excel('Poster_Figures/Run 1 - Rise times and Amplitude aligned/EPSCs_Dataframe_normal_1000traces_lognormalglut_1000channels.xlsx')
    # folder_name = 'Poster_Figures/Run 1 - Rise times and Amplitude aligned/'
    # sim_raw_rise_times = rise_times_histogram_creator(EPSCS,folder_name =folder_name,plt_show=True)
    # sim_raw_taus = tau_graph_generator(EPSCS,folder_name =folder_name,plt_show = True )
    # sim_raw_amplitudes = amplitude_histogram_creator(EPSCS,folder_name =folder_name,plt_show = True)
    #
    #
    #
    #
    #
    #
    #
    #
    # for x, y in zip(sim_raw_taus, sim_raw_amplitudes):
    #
    #     scatter = plt.scatter(x, y, color='black')
    #
    # plt.xlabel('Decay Taus')
    # plt.ylabel('Max Amplitude (pA)')
    # plt.title("Simulated")
    #
    # plt.show()
    #
    # for x, y in zip(sim_raw_rise_times, sim_raw_amplitudes):
    #
    #     scatter = plt.scatter(x, y, color='black')
    # plt.xlabel('Rise Times (ms)')
    # plt.ylabel('Max Amplitude (pA)')
    # plt.title("Simulated")
    #
    # plt.show()
    #
    #
    #
    #
    #
    #

    # file_name = 'data-files/Many_EPSCs_Juan.xlsx'
    # # EPSCs = pd.read_excel(file_name)
    # folder_name = 'Many_EPSCs_Juan/'
    #
    # # # ##Many CDF generation
    # # # file_name = 'IEM_Data/IEM_EPSCs.xlsx'
    # # file_name = "data-files/EPSCs_Dataframe_200Channels_200traces_CICP.xlsx"
    # exp_rise_times = []
    # exp_amplitudes = []
    # exp_taus = []
    # sheet_names = []
    # # folder_name = 'IEM_Data/'
    # # folder_name = 'Simulated_Data_ARO'
    # #####Experimental values generation#####
    # for sheet in tqdm(range(0,7)):
    #     EPSCs = pd.read_excel(file_name, sheet_name=sheet)
    #     excel_file = pd.ExcelFile(file_name)
    #     all_sheet_names = excel_file.sheet_names
    #     sheet_name = all_sheet_names[sheet]
    #     sheet_names.append(sheet_name)
    #
    #     if not os.path.exists(folder_name):
    #         os.makedirs(folder_name)
    #
    #     max_amplitudes = amplitude_histogram_creator(EPSCs, folder_name, False)
    #     rise_times = rise_times_histogram_creator(EPSCs, folder_name, False)
    #     print(len(rise_times))
    #     taus_array = tau_graph_generator(EPSCs, folder_name, False)
    #
    #     exp_rise_times.append(rise_times)
    #     exp_amplitudes.append(max_amplitudes)
    #     exp_taus.append(taus_array)
    # print(exp_taus)
    # colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    #
    # for i, (rise_times, taus) in enumerate(zip(exp_rise_times, exp_taus)):
    #     plt.scatter(taus, rise_times, color=colors[i], label=sheet_names[i])
    #
    # plt.ylabel("Rise times (ms)")
    # plt.xlabel("Taus")
    # plt.legend()  # Add legend
    # plt.show()
    # colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    # for i, (rise_times, taus) in enumerate(zip(exp_rise_times, exp_taus)):
    #     plt.scatter(taus, rise_times, color=colors[i], label=sheet_names[i])
    #
    # plt.ylabel("Rise times (ms)")
    # plt.xlabel("Taus")
    # plt.legend()  # Add legend
    # plt.show()
    #
    # # Compute means
    # mean_rise_times = [np.mean(rise_times) for rise_times in exp_rise_times]
    # mean_taus = [np.mean(taus) for taus in exp_taus]
    #
    # # Define colors
    # colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    #
    # # Scatter plot of means
    # plt.scatter(mean_taus, mean_rise_times, color=colors, label=sheet_names)
    #
    # # Add labels to each point
    # for i, (x, y) in enumerate(zip(mean_rise_times, mean_taus)):
    #     plt.text(y, x, sheet_names[i], fontsize=9, ha='right')
    #
    # plt.xlabel("Mean Rise Times")
    # plt.ylabel("Mean Taus")
    # plt.title("Scatter Plot of Mean Rise Times vs. Mean Taus")
    # plt.show()
    # multiple_cdf_generator(exp_rise_times,labels=sheet_names,folder_name=folder_name,plt_show=True, plt_type="IEM ",data_type="Rise Times",unit="(ms)")
    # multiple_cdf_generator(exp_amplitudes,labels=sheet_names,folder_name=folder_name,plt_show=True, plt_type="IEM ",data_type="Max Amplitude",unit="(pA)")
    # multiple_cdf_generator(exp_taus,labels=sheet_names,folder_name=folder_name,plt_show=True, plt_type="IEM ", data_type = "Decay Taus")

    # ######Simulation Info Generator ###########
    # # Define the folder containing the text files
    # folder_name = "Simulated_Data_ARO/Uniform_1000channels/"
    #
    # # Find all excel files in the folder
    # excel_files = glob.glob(os.path.join(folder_name, "*.xlsx"))
    #
    # sim_rise_times = []
    # sim_amplitudes = []
    # sim_taus = []
    # file_names = []
    # # Load each text file into a separate DataFrame
    # for file in tqdm(excel_files, desc="Loading Excel files", unit="file"):
    #     EPSCs = pd.read_excel(file)  # Load data from text file
    #     # EPSC_Plot(EPSCs,folder_name,True)
    #     max_amplitudes = amplitude_histogram_creator(EPSCs,folder_name,False)
    #     rise_times = rise_times_histogram_creator(EPSCs,folder_name, False)
    #     taus_array = tau_graph_generator(EPSCs,folder_name,False)
    #     # tau_amplitude_scatterplot(taus_array,max_amplitudes,folder_name, plt_show =True)
    #     # rise_time_amplitude_scatterplot(rise_times,max_amplitudes,folder_name,True)
    #     #cdf_generator(rise_times,folder_name,plt_show=False)
    #     sim_rise_times.append(rise_times)
    #     sim_amplitudes.append(max_amplitudes)
    #     sim_taus.append(taus_array)
    #     file_name = os.path.basename(file)
    #     file_names.append(file_name)
    # # sheet_names = ["1mM","10mM","100mM"]
    # labels = ['100mM', '10mM', '1mM']
    #
    # # ks_matrix_generator(exp_taus,sim_taus,title="Decay Taus")
    # # ks_matrix_generator(exp_rise_times,sim_rise_times,title="Rise Times")
    # # ks_matrix_generator(exp_amplitudes,sim_amplitudes,title="Max Amplitudes")
    #
    # multiple_cdf_generator(sim_rise_times,labels=labels,folder_name=folder_name,plt_show=True, plt_type="Simulation ",data_type="Rise Times",unit="(ms)")
    # multiple_cdf_generator(sim_amplitudes,labels=labels,folder_name=folder_name,plt_show=True, plt_type="Simulation ",data_type="Max Amplitudes",unit="(pa)")
    # multiple_cdf_generator(sim_taus,labels=labels,folder_name=folder_name,plt_show=True, plt_type="Simulation ", data_type ="Decay Taus")
    #
    # #plot the rise times colored by their concentration
    # data = sim_rise_times
    # df_long = pd.DataFrame({
    #     'Concentration': np.repeat(labels, [len(group) for group in data]),
    #     'Time(ms)': np.concatenate(data)
    # })
    # # Create the plot
    # plt.figure(figsize=(10, 6))
    #
    # # Add boxplot first (with reduced opacity)
    # sns.boxplot(x='Concentration', y='Time(ms)', data=df_long,
    #             width=0.5,
    #             color='lightgray',
    #             )  # Make boxplot semi-transparent
    #
    # # Overlay stripplot
    # sns.stripplot(x='Concentration', y='Time(ms)', data=df_long,
    #               jitter=True,  # Spread out points
    #               alpha=0.7,  # Slightly transparent points
    #               color='blue',  # Optional: choose a specific color
    #               size=5)  # Adjust point size if needed
    #
    # # Customize the plot
    # plt.title('Glutamate concentration vs Rise Time')
    # plt.xlabel('Concentration [mM]')
    # plt.ylabel('Time(ms)')
    #
    # # Adjust layout and display
    # plt.tight_layout()
    # plt.savefig(f"{folder_name}strip_plot.png")
    # plt.show()

if __name__ == "__main__":
    main()





##########################Graveyard
    #If using Jim's simulated data:
    # EPSCs = np.loadtxt('data-files/Simulated_EPSCs_CPCI.txt', dtype=float)  # Specify dtype=float
    # EPSCs = pd.DataFrame(EPSCs)
    # folder_name = 'VB_Simulated_EPSCs_CPCI/'

    #For Juan's Experimental
    # EPSCs = np.loadtxt('data-files/Experimental_EPSCs_200.txt', dtype=float)  # Specify dtype=float
    # EPSCs = pd.DataFrame(EPSCs)
    # folder_name = 'Experimental_200_plts/'

    # For Juan's (Many) Experimental 1/28
    # file_name = 'data-files/Many_EPSCs_Juan.xlsx'
    # EPSCs = pd.read_excel(file_name)
    # folder_name = 'Many_EPSCs_Juan/'

    #For 200-1000-Mixed Current
    # file_name = 'data-files/EPSCs_Dataframe_200Channels_200traces_CICP.xlsx'
    # EPSCs = pd.read_excel(file_name)
    # folder_name = 'PythonSim-200Channels-200traces-CICP/'

    #For 500-1000
    # file_name = 'data-files/EPSCs_Dataframe_500channels.xlsx'
    # EPSCs = pd.read_excel(file_name)
    # folder_name = '500_channels/'

    #For recent simulation VB output (4 traces)
    # file_name = 'data-files/vb-4traces.xlsx'
    # EPSCs = pd.read_excel(file_name,sheet_name="Sheet2")
    # folder_name = 'vb-4traces/'