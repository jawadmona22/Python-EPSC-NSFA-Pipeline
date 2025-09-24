import os
from Scripts.EPSC_Simulation.EPSC_Simulation import EPSC_Calc
import pandas as pd
import matplotlib.pyplot as plt
from Scripts.NSFA_Tools.EPSC_Matrix_Generator import matrix_generator
import Scripts.NSFA_Tools.EPSC_Graphic_Utilities as egg
import math
from sklearn.cluster import KMeans
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import ast
import seaborn as sns
"""The purpose of this experiment is to 
Understand how changing the fixed/non-uniform distribution of channels and [Glutamate] influence the shape of NSFA 
graphs and the extraction of (i) and (n) for simulated EPSC data. """


def create_fixed_simulation():

    fixed_folder_path = "C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\Scripts\\Experiments\\Simulation_Glu_N_Settings_Investigation\\fixed_EPSCs_gl1_ch200.xlsx"
    fixed_glutamate_params = pd.DataFrame(
        [{"gl_mean": 0, "gl_sd": 0, "distribution_type": "fixed_value", "fixed_value": 1}])

    fixed_channel_params = pd.DataFrame(
        [{"distribution_type": "fixed_value", "channel_sd": 0, "channel_mean": 0, "fixed_value": 200}])


    EPSCs_df, all_total_channel_nums, channel_data_df = EPSC_Calc(num_EPSCs=200,channel_params=fixed_channel_params,glutamate_params=fixed_glutamate_params, output_file_path=fixed_folder_path)

def create_normn_fixedglu_simulation():

    normn_fixedglu_folder_path = "C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\Scripts\\Experiments\\Simulation_Glu_N_Settings_Investigation\\normn_fixedglu.xlsx"
    fixed_glutamate_params = pd.DataFrame(
        [{"gl_mean": 0, "gl_sd": 0, "distribution_type": "fixed_value", "fixed_value": 3.5}])

    norm_channel_params = pd.DataFrame(
        [{"distribution_type": "normal", "channel_sd": 560, "channel_mean": 1800, "fixed_value": None}])


    EPSCs_df, all_total_channel_nums, channel_data_df = EPSC_Calc(num_EPSCs=1000,channel_params=norm_channel_params,glutamate_params=fixed_glutamate_params, output_file_path=normn_fixedglu_folder_path)

def create_normaln_lognglu_simulation():

    normaln_lognglu_folder_path = "C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\Scripts\\Experiments\\Simulation_Glu_N_Settings_Investigation\\normaln_lognglu.xlsx"
    logn_glutamate_params = pd.DataFrame(
        [{"gl_mean": 3.5,"gl_sd": 1, "distribution_type": "lognormal", "fixed_value": None}])

    norm_channel_params = pd.DataFrame(
        [{"distribution_type": "normal", "channel_sd": 560, "channel_mean": 1800, "fixed_value": None}])


    EPSCs_df, all_total_channel_nums, channel_data_df = EPSC_Calc(num_EPSCs=1000,channel_params=norm_channel_params,glutamate_params=logn_glutamate_params, output_file_path=normaln_lognglu_folder_path)

def create_fixedn_lognglu_simulation():

    fixedn_lognglu_folder_path = "C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\Scripts\\Experiments\\Simulation_Glu_N_Settings_Investigation\\fixedn_lognglu.xlsx"
    logn_glutamate_params = pd.DataFrame(
        [{"gl_mean": 3.5,"gl_sd": 1, "distribution_type": "lognormal", "fixed_value": None}])

    fixed_channel_params = pd.DataFrame(
        [{"distribution_type": "fixed_value", "channel_sd": 0, "channel_mean": 0, "fixed_value": 1000}])


    EPSCs_df, all_total_channel_nums, channel_data_df = EPSC_Calc(num_EPSCs=1000,channel_params=fixed_channel_params,glutamate_params=logn_glutamate_params, output_file_path=fixedn_lognglu_folder_path)


def NSFA_Simulation_Settings():
    ##Create a combined matrix
    output_file = 'simulationsettings_epscs.xlsx'
    folder_path = "C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\Scripts\\Experiments\\Simulation_Glu_N_Settings_Investigation\\EPSC_Data"

    with pd.ExcelWriter(output_file) as writer:
        for file in os.listdir(folder_path):
            if file.endswith(".xlsx"):
                file_path = os.path.join(folder_path, file)
                print(f"Processing file {file_path}")
                df = pd.read_excel(file_path,sheet_name=0,engine='openpyxl')
                # Write the file to an Excel sheet
                # Remove xlsx
                df.to_excel(writer, sheet_name=str(file), index=False,
                            header=False)  # 7: is to make the name <=31 chars

    params = {
        "alignment": ["peak"],
        "direct_df_input":False,
        "analysis_start_point": ["peak_start"],
        "scaling": ["minimize_error"],
        "output": ["linear", "parabolic"],
        "file_name": 'simulationsettings_epscs.xlsx',
        "folder_name": "C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\Scripts\\Experiments\\Simulation_Glu_N_Settings_Investigation"
    }

    matrix = matrix_generator(params,first_sheet=False)
    matrix_df = pd.DataFrame(matrix)
    matrix_df.to_excel('simulationsettings_matrix.xlsx')


def Check_CDF_Comparison():
#Simulations with .5 managed to give us a parabolic return. But how close are they to physiological cell CDF?
    print("Checking CDF comparison...")

def Rise_Times_Simulation_Settings():
    print("Beginning rise time/amplitude analysis")
    folder_path = "C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\Scripts\\Experiments\\Simulation_Glu_N_Settings_Investigation\\EPSC_Data"
    for file in os.listdir(folder_path):
        if file.endswith(".xlsx"):
            file_path = os.path.join(folder_path, file)
            print(f"Creating histogram for file {file_path}")
            simulation_EPSCs = pd.read_excel(file_path,sheet_name=0)
            simulation_rise_times = egg.rise_times_histogram_creator(simulation_EPSCs, folder_name=folder_path,
                                                                     plt_show=False)
            simulation_amplitudes = egg.amplitude_histogram_creator(simulation_EPSCs, folder_name=folder_path,
                                                                    plt_show=False)
            plt.close('all')
            plt.scatter(simulation_rise_times,simulation_amplitudes)
            plt.title(f"Rise Times vs. Amplitude for {file[:-4]}")
            plt.xlabel("Rise Times (ms)")
            plt.ylabel("Max Amplitudes (pA)")
            plt.xlim(0, .4)
            plt.ylim(0, 1000)

            plt.show()

def Rise_Time_Physiological(): #Checking the histogram for a specific cell...
    file_path = "C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\data-files\\Many_EPSCs_Juan.xlsx"
    folder_path = "C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\Scripts\\Experiments\\Simulation_Glu_N_Settings_Investigation"
    # Load all sheet names
    xls = pd.ExcelFile(file_path)
    sheet_names = xls.sheet_names
    num_sheets = len(sheet_names)

    # Determine subplot grid size (e.g., 3x3 for 9 sheets)
    cols = 3
    rows = math.ceil(num_sheets / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    fig.tight_layout()

    axes = axes.flatten()  # Flatten in case of multiple rows

    for idx, sheet in enumerate(sheet_names):
        print(f"Reading sheet: {sheet}")
        simulation_EPSCs = pd.read_excel(xls, sheet_name=sheet)

        simulation_rise_times = egg.rise_times_histogram_creator(simulation_EPSCs, folder_name=folder_path,
                                                                 plt_show=False)
        simulation_amplitudes = egg.amplitude_histogram_creator(simulation_EPSCs, folder_name=folder_path,
                                                                plt_show=False)

        ax = axes[idx]
        ax.scatter(simulation_rise_times, simulation_amplitudes,label=f'{sheet}')
        ax.legend()
        ax.set_xlim(0, 0.4)
        ax.set_ylim(0, 1000)
    ax.set_xlabel("Rise Times (ms)")
    ax.set_ylabel("Max Amplitudes (pA)")
    # Hide any unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    # fig.suptitle("Rise Times vs Amplitudes Across Sheets", fontsize=16)
    plt.show()


def find_elbow(k_vals, errors):
    # Normalize to [0,1]
    k_norm = (k_vals - np.min(k_vals)) / (np.max(k_vals) - np.min(k_vals))
    e_norm = (errors - np.min(errors)) / (np.max(errors) - np.min(errors))

    # Line from first to last point
    line = np.array([k_norm[0], e_norm[0]]), np.array([k_norm[-1], e_norm[-1]])

    # Compute distances to the line
    line_vec = line[1] - line[0]
    distances = []
    for i in range(len(k_vals)):
        point = np.array([k_norm[i], e_norm[i]])
        proj_len = np.dot(point - line[0], line_vec) / np.dot(line_vec, line_vec)
        proj = line[0] + proj_len * line_vec
        dist = np.linalg.norm(point - proj)
        distances.append(dist)
    return k_vals[np.argmax(distances)]
def RiseTime_Amplitude_Clustering():
    file_path = "C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\data-files\\Many_EPSCs_Juan.xlsx"
    folder_path = "C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\Scripts\\Experiments\\Simulation_Glu_N_Settings_Investigation"
    # Load all sheet names
    xls = pd.ExcelFile(file_path)
    sheet_names = xls.sheet_names
    num_sheets = len(sheet_names)

    # Determine subplot grid size (e.g., 3x3 for 9 sheets)
    cols = 3
    rows = math.ceil(num_sheets / cols)


    fig_clust, axes_clust = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    fig_clust.tight_layout()

    axes_clust = axes_clust.flatten()

    fig_el, ax_el = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    fig_clust.tight_layout()

    ax_el = ax_el.flatten()

    for idx, sheet in enumerate(sheet_names):
            print(f"Reading sheet: {sheet}")
            simulation_EPSCs = pd.read_excel(xls, sheet_name=sheet)

            simulation_rise_times = egg.rise_times_histogram_creator(simulation_EPSCs, folder_name=folder_path,
                                                                     plt_show=False)
            simulation_amplitudes = egg.amplitude_histogram_creator(simulation_EPSCs, folder_name=folder_path,
                                                                    plt_show=False)
            ##Try some clustering
            X = np.column_stack((simulation_rise_times, simulation_amplitudes))
            k_range = range(1,10)
            inertias = []

            for k in k_range:
                kmeans = KMeans(n_clusters=k,random_state=0)
                kmeans.fit(X)
                inertias.append(kmeans.inertia_)

            optimal_k = find_elbow(np.array(list(k_range)),np.array(inertias))
            print(f"Automatically selected optimal number of clusters: {optimal_k}")
            # Step 3: Final KMeans with optimal k
            kmeans = KMeans(n_clusters=optimal_k, random_state=0)
            kmeans.fit(X)
            labels = kmeans.labels_
            original_labels = kmeans.labels_

            centroids = kmeans.cluster_centers_

            sorted_indices = sorted(range(optimal_k), key=lambda i: centroids[i][1])  # sort by y

            # Create a mapping from original label → new label (ranked by y)
            label_map = {old: new for new, old in enumerate(sorted_indices)}

            # Apply the mapping to the cluster labels
            relabelled = np.vectorize(label_map.get)(original_labels)

            # Sort centroids accordingly (for plotting)
            sorted_centroids = centroids[sorted_indices]

            # Define custom colormap from lowest to highest
            custom_cmap = ListedColormap(['yellow', 'purple', 'blue'])  # can extend if needed

            ax = axes_clust[idx]

            ax.scatter(X[:, 0], X[:, 1], c=relabelled, cmap=custom_cmap, alpha=0.7,label=sheet)
            ax.scatter(sorted_centroids[:, 0], sorted_centroids[:, 1], c='black', marker='X', s=100)
            ax.set_xlim(0, 0.4)
            ax.set_ylim(0, 1000)
            ax.legend()

            el = ax_el[idx]
            el.plot(k_range, inertias, marker='o')
            el.axvline(optimal_k, color='red', linestyle='--')
            el.set_title(sheet, fontsize=10)
            el.set_xlabel('k')
            el.set_ylabel('Inertia')



    plt.show()

def RiseTime_Simulation_Clustering(): #Checking the histogram for a specific cell...
    folder_path = "C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\Scripts\\Experiments\\Simulation_Glu_N_Settings_Investigation\\EPSC_Data"

    # Determine subplot grid size (e.g., 3x3 for 9 sheets)
    cols = 3
    num_sheets = 10
    rows = math.ceil(num_sheets / cols)


    fig_clust, axes_clust = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    fig_clust.tight_layout()

    axes_clust = axes_clust.flatten()

    fig_el, ax_el = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    fig_clust.tight_layout()

    ax_el = ax_el.flatten()
    idx = 0
    for file in os.listdir(folder_path):
        if file.endswith(".xlsx"):
            file_path = os.path.join(folder_path, file)
            print(f"Creating histogram for file {file_path}")
            simulation_EPSCs = pd.read_excel(file_path, sheet_name=0)
            simulation_rise_times = egg.rise_times_histogram_creator(simulation_EPSCs, folder_name=folder_path,
                                                                     plt_show=False)
            simulation_amplitudes = egg.amplitude_histogram_creator(simulation_EPSCs, folder_name=folder_path,
                                                                    plt_show=False)


            ##Try some clustering
            X = np.column_stack((simulation_rise_times, simulation_amplitudes))
            k_range = range(1,10)
            inertias = []

            for k in k_range:
                kmeans = KMeans(n_clusters=k,random_state=0)
                kmeans.fit(X)
                inertias.append(kmeans.inertia_)

            optimal_k = find_elbow(np.array(list(k_range)),np.array(inertias))
            print(f"Automatically selected optimal number of clusters: {optimal_k}")
            # Step 3: Final KMeans with optimal k
            kmeans = KMeans(n_clusters=optimal_k, random_state=0)
            kmeans.fit(X)
            labels = kmeans.labels_
            original_labels = kmeans.labels_

            centroids = kmeans.cluster_centers_

            sorted_indices = sorted(range(optimal_k), key=lambda i: centroids[i][1])  # sort by y

            # Create a mapping from original label → new label (ranked by y)
            label_map = {old: new for new, old in enumerate(sorted_indices)}

            # Apply the mapping to the cluster labels
            relabelled = np.vectorize(label_map.get)(original_labels)

            # Sort centroids accordingly (for plotting)
            sorted_centroids = centroids[sorted_indices]

            # Define custom colormap from lowest to highest
            custom_cmap = ListedColormap(['yellow', 'purple', 'blue'])  # can extend if needed

            ax = axes_clust[idx]

            ax.scatter(X[:, 0], X[:, 1], c=relabelled, cmap=custom_cmap, alpha=0.7,label=file[:-4])
            ax.scatter(sorted_centroids[:, 0], sorted_centroids[:, 1], c='black', marker='X', s=100)
            ax.set_xlim(0, 0.4)
            ax.set_ylim(0, 1000)
            ax.legend()

            el = ax_el[idx]
            el.plot(k_range, inertias, marker='o')
            el.axvline(optimal_k, color='red', linestyle='--')
            el.set_xlabel('k')
            el.set_ylabel('Inertia')
            idx +=1



    plt.show()

def Phys_DBSCAN():


    # Setup
    file_path = "C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\data-files\\Many_EPSCs_Juan.xlsx"
    folder_path = "C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\Scripts\\Experiments\\Simulation_Glu_N_Settings_Investigation"
    xls = pd.ExcelFile(file_path)
    sheet_names = xls.sheet_names

    # Subplot grid
    n_sheets = len(sheet_names)
    n_cols = 4
    n_rows = math.ceil(n_sheets / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows), squeeze=False)
    axes = axes.flatten()

    for i, sheet in enumerate(sheet_names):
        try:
            df = pd.read_excel(xls, sheet_name=sheet)

            # Use your feature functions
            rise_times = egg.rise_times_histogram_creator(df, folder_name=folder_path, plt_show=False)
            amplitudes = egg.amplitude_histogram_creator(df, folder_name=folder_path, plt_show=False)
            rise_times = np.array(rise_times)
            amplitudes = np.array(amplitudes)
            X = np.column_stack((rise_times, amplitudes))
            X_scaled = StandardScaler().fit_transform(X)

            # DBSCAN parameters — tune these if needed
            db = DBSCAN(eps=.4, min_samples=10).fit(X_scaled)
            labels = db.labels_

            # Plot
            ax = axes[i]
            unique_labels = set(labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

            for label, color in zip(unique_labels, colors):
                mask = (labels == label)
                if label == -1:
                    # Noise
                    color = 'gray'
                    label_name = 'Noise'
                else:
                    label_name = f'Cluster {label}'
                ax.scatter(rise_times[mask], amplitudes[mask], s=10, color=color, label=label_name, alpha=0.7)

            ax.set_title(sheet, fontsize=10)
            ax.set_xlabel('Rise Time (ms)')
            ax.set_ylabel('Amplitude (pA)')
            ax.set_xlim(0, 0.4)
            ax.set_ylim(0, 1000)
            ax.legend(fontsize=6, loc='upper right')
            ax.grid(True)

        except Exception as e:
            print(f"Error processing {sheet}: {e}")
            axes[i].set_visible(False)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.suptitle('DBSCAN Clustering of EPSC Rise Time vs Amplitude', fontsize=16, y=1.02)
    plt.subplots_adjust(top=0.95)
    plt.show()

def Rise_Times_Simulation_Colored_Glu():
    print("Beginning rise time/amplitude analysis")
    folder_path = "C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\Scripts\\Experiments\\Simulation_Glu_N_Settings_Investigation\\EPSC_Data"
    cmap = cm.get_cmap('tab20', 20)  # 40 samples for smooth interpolation
    label_colors = {i: cmap(i / 19) for i in np.arange(0, 10, 0.5)}

    for file in os.listdir(folder_path):
        if file.endswith(".xlsx"):
            file_path = os.path.join(folder_path, file)
            print(f"Creating histogram for file {file_path}")
            simulation_EPSCs = pd.read_excel(file_path,sheet_name=0)
            channel_data = pd.read_excel(file_path,sheet_name="Channel Data")
            glut_data = channel_data['Glutamate Concentration']
            simulation_rise_times = egg.rise_times_histogram_creator(simulation_EPSCs, folder_name=folder_path,
                                                                     plt_show=False)
            simulation_amplitudes = egg.amplitude_histogram_creator(simulation_EPSCs, folder_name=folder_path,
                                                                    plt_show=False)
            plt.close('all')
            print(len(simulation_amplitudes))
            print(len(simulation_rise_times))
            print(glut_data.shape)

            df = pd.DataFrame({'RiseTimes':simulation_rise_times,'Amplitudes':simulation_amplitudes,'Glu':channel_data['Glutamate Concentration'].values})

            fallback_cmap = cm.get_cmap('viridis')
            df['Glu'] = df['Glu'].apply(lambda x: float(ast.literal_eval(x)[0]) if isinstance(x, str) else float(x))

            norm = mcolors.Normalize(vmin=df['Glu'].min(), vmax=df['Glu'].max())

            sc = None

            for label, group in df.groupby('Glu'):
                if label in label_colors:
                    color = label_colors[label]
                    plt.scatter(group['RiseTimes'], group['Amplitudes'], color=color)
                else:
                    # Use continuous colormap
                    sc = plt.scatter(group['RiseTimes'], group['Amplitudes'],
                                     c=[label] * len(group), cmap=fallback_cmap, norm=norm,
                                     label=f'{label}')

            # Add colorbar if any point used the continuous colormap
            if sc is not None:
                cbar = plt.colorbar(sc)
                cbar.set_label("Glutamate Concentration (mM)")
            else:
                plt.legend(title="Glutamate Concentration (mM)")
            # Add common plot elements
            plt.title(f"Rise Times vs. Amplitude for {file[:-4]}")
            plt.xlabel("Rise Times (ms)")
            plt.ylabel("Max Amplitudes (pA)")
            plt.xlim(0, 0.4)
            plt.ylim(0, 1000)

            # Save or show the completed plot
            plt.savefig(f"{file[:-4]}.png")
            plt.show()

def create_normn_normglu_continuous_data(): #Realized here that lognorm

    normaln_lognglu_folder_path = "C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\Scripts\\Experiments\\Simulation_Glu_N_Settings_Investigation\\normaln_normalglu_c.xlsx"
    logn_glutamate_params = pd.DataFrame(
        [{"gl_mean": 3.5,"gl_sd": 1, "distribution_type": "normal", "fixed_value": None,"continuous":True}])

    norm_channel_params = pd.DataFrame(
        [{"distribution_type": "normal", "channel_sd": 560, "channel_mean": 1800, "fixed_value": None}])


    EPSCs_df, all_total_channel_nums, channel_data_df = EPSC_Calc(num_EPSCs=1000,channel_params=norm_channel_params,glutamate_params=logn_glutamate_params, output_file_path=normaln_lognglu_folder_path)

def Rise_Times_Simulation_Colored_N():

        print("Beginning rise time/amplitude analysis")
        folder_path = "C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\Scripts\\Experiments\\Simulation_Glu_N_Settings_Investigation\\EPSC_Data"


        for file in os.listdir(folder_path):
            if file.endswith(".xlsx"):
                file_path = os.path.join(folder_path, file)
                print(f"Creating histogram for file {file_path}")
                simulation_EPSCs = pd.read_excel(file_path, sheet_name=0)
                channel_data = pd.read_excel(file_path, sheet_name="Channel Data")
                open_channels = channel_data['Open Channels']
                simulation_rise_times = egg.rise_times_histogram_creator(simulation_EPSCs, folder_name=folder_path,
                                                                         plt_show=False)
                simulation_amplitudes = egg.amplitude_histogram_creator(simulation_EPSCs, folder_name=folder_path,
                                                                        plt_show=False)
                plt.close('all')
                print(len(simulation_amplitudes))
                print(len(simulation_rise_times))
                print(open_channels.shape)

                df = pd.DataFrame({'RiseTimes': simulation_rise_times, 'Amplitudes': simulation_amplitudes,
                                   'open_channels': open_channels.values})

                fallback_cmap = cm.get_cmap('viridis')
                df['open_channels'] = df['open_channels'].apply(lambda x: float(ast.literal_eval(x)[0]) if isinstance(x, str) else float(x))

                norm = mcolors.Normalize(vmin=100, vmax=1500)

                sc = None

                for label, group in df.groupby('open_channels'):

                    # Use continuous colormap
                    sc = plt.scatter(group['RiseTimes'], group['Amplitudes'],
                                     c=[label] * len(group), cmap=fallback_cmap, norm=norm,
                                     label=f'{label}')

                # Add colorbar if any point used the continuous colormap
                if sc is not None:
                    cbar = plt.colorbar(sc)
                    cbar.set_label("Open Channels")
                # Add common plot elements
                plt.title(f"Rise Times vs. Amplitude for {file[:-4]}")
                plt.xlabel("Rise Times (ms)")
                plt.ylabel("Max Amplitudes (pA)")
                plt.xlim(0, 0.4)
                plt.ylim(0, 1000)

                # Save or show the completed plot
                plt.savefig(f"{file[:-4]}-n.png")
                plt.show()

def RiseTime_Amplitude_Clustering_IEM(): #Compare RT/Amplitude for Control/IEM
    iem_file_path = "C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\data-files\\IEM_Data\\IEM_EPSCs.xlsx"
    ctrl_file_path = "C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\data-files\\Many_EPSCs_Juan.xlsx"

    folder_path = "C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\Scripts\\Experiments\\Simulation_Glu_N_Settings_Investigation"
    # Load all sheet names
    xls = pd.ExcelFile(ctrl_file_path)

    iem_xls = pd.ExcelFile(iem_file_path)

    iem_sheet_names = iem_xls.sheet_names

    sheet_names = xls.sheet_names
    num_sheets = len(sheet_names)

    # Determine subplot grid size
    cols = 2
    rows = math.ceil(num_sheets)


    fig_clust, axes_clust = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    fig_clust.tight_layout()

    axes_clust = axes_clust.flatten()

    fig_clust.tight_layout()





    for idx, sheet in enumerate(sheet_names):
            print(f"Reading sheet: {sheet}")

            #Run Control
            simulation_EPSCs = pd.read_excel(xls, sheet_name=sheet)

            simulation_rise_times = egg.rise_times_histogram_creator(simulation_EPSCs, folder_name=folder_path,
                                                                     plt_show=False)
            simulation_amplitudes = egg.amplitude_histogram_creator(simulation_EPSCs, folder_name=folder_path,
                                                                    plt_show=False)


            ax = axes_clust[idx*2]

            ax.scatter(simulation_rise_times, simulation_amplitudes,alpha=0.7,label=sheet)
            # ax.scatter(sorted_centroids[:, 0], sorted_centroids[:, 1], c='black', marker='X', s=100)
            ax.set_xlim(0, 0.4)
            ax.set_ylim(0, 1000)
            ax.legend()

            #-----------Run IEM---------------#



            #find index where (if) matching IEM data lies
            match_index = -1
            iem_matching_sheet = None
            for index,iem_sheet in enumerate(iem_sheet_names):
                if iem_sheet.split('_')[0] == sheet:
                    match_index = index
                    iem_matching_sheet = iem_sheet

            if match_index != -1:
                iem_simulation_EPSCs = pd.read_excel(iem_xls, sheet_name=iem_matching_sheet)

                iem_simulation_rise_times = egg.rise_times_histogram_creator(iem_simulation_EPSCs,
                                                                             folder_name=folder_path,
                                                                             plt_show=False)
                iem_simulation_amplitudes = egg.amplitude_histogram_creator(iem_simulation_EPSCs,
                                                                            folder_name=folder_path,
                                                                            plt_show=False)

                ax = axes_clust[(2*idx)+1]

                ax.scatter(iem_simulation_rise_times, iem_simulation_amplitudes, alpha=0.7, label=iem_matching_sheet)
                ax.set_xlim(0, 0.4)
                ax.set_ylim(0, 1000)
                ax.legend()



def NSFA_continous_optimized():

    ##Create a combined matrix
    folder_path = "C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\Scripts\\Experiments\\Simulation_Glu_N_Settings_Investigation\\EPSC_Data\\EPSCS_3XSim_100_CDFfit.xlsx"
    epscs = pd.read_excel(folder_path)
    params = {
        "alignment": ["peak"],
        "direct_df_input": True,
        "analysis_start_point": ["peak_start"],
        "scaling": ["minimize_error"],
        "output": ["linear", "parabolic"],
        "file_name": 'cont-opt-epscs.xlsx',
        "folder_name": "C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\Scripts\\Experiments\\Simulation_Glu_N_Settings_Investigation",
        "sheet_names": [0],
        "EPSCs":epscs
    }


    matrix = matrix_generator(params, first_sheet=True)
    matrix_df = pd.DataFrame(matrix)
    matrix_df.to_excel('cont-opt-matrix.xlsx')


def NSFA_continous_optimized_all_matrix_options():

    ##Create a combined matrix
    folder_path = "C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\Scripts\\Experiments\\Simulation_Glu_N_Settings_Investigation\\EPSC_Data\\EPSCS_3XSim_100_CDFfit.xlsx"
    epscs = pd.read_excel(folder_path)
    channel_data = pd.read_excel(folder_path, sheet_name="Channel Data")
    mean_channels = channel_data["Open Channels"].mean()
    params = {
        "alignment":["peak","midpoint","max_dv_dt"],
        "direct_df_input": True,
        "analysis_start_point":["peak_start"],
        "scaling":["minimize_error","peak_scaling_at_peak_time","peak_to_peak_scaling"],
        "output": ["linear","parabolic"],
        "file_name": 'cont-opt-epscs.xlsx',
        "folder_name": "C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\Scripts\\Experiments\\Simulation_Glu_N_Settings_Investigation\\NSFA_Output_Opt",
        "sheet_names": [0],
        "EPSCs":epscs,
        "mean_channels": mean_channels,
        "recording_duration":16
    }


    matrix = matrix_generator(params, first_sheet=True,debug=True)
    matrix_df = pd.DataFrame(matrix)
    matrix_df.to_excel('cont-opt-matrix-many.xlsx')

def NSFA_control_all_options():

    ##Create a combined matrix
    folder_path = "C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\data-files\\test.xlsx"
    epscs = pd.read_excel(folder_path,sheet_name=0)
    channel_data = pd.read_excel(folder_path, sheet_name="Channel Data")
    mean_channels = channel_data["Open CI Channels"].mean()
    params = {
        "alignment":["peak","midpoint","max_dv_dt"],
        "direct_df_input": True,
        "analysis_start_point":["peak_start",],
        "scaling":["minimize_error","peak_scaling_at_peak_time","peak_to_peak_scaling","raw"],
        "output": ["linear","parabolic"],
        "file_name": folder_path,
        "folder_name": "C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\Scripts\\Experiments\\Simulation_Glu_N_Settings_Investigation\\NSFA_Output_Control_Test",
        "sheet_names": ["Control"],
        "EPSCs":epscs,
        "mean_channels":mean_channels,
        "recording_duration":16
    }


    matrix = matrix_generator(params, first_sheet=True)
    matrix_df = pd.DataFrame(matrix)
    matrix_df.to_excel('control-matrix-test.xlsx')

def Amplitude_Decay_Simulation_Analysis():

        print("Beginning decay/amplitude analysis")
        folder_path = "C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\Scripts\\Experiments\\Simulation_Glu_N_Settings_Investigation\\EPSC_Data"

        for file in os.listdir(folder_path):
            if file.endswith(".xlsx"):
                file_path = os.path.join(folder_path, file)
                print(f"Creating histogram for file {file_path}")
                simulation_EPSCs = pd.read_excel(file_path, sheet_name=0)
                channel_data = pd.read_excel(file_path, sheet_name="Channel Data")
                open_channels = channel_data['Open Channels']
                simulation_decay_taus = egg.tau_graph_generator(simulation_EPSCs, folder_name=folder_path,
                                                                         plt_show=False,time=16)
                simulation_amplitudes = egg.amplitude_histogram_creator(simulation_EPSCs, folder_name=folder_path,
                                                                        plt_show=False)
                plt.close('all')
                print(len(simulation_amplitudes))
                print(len(simulation_decay_taus))
                print(open_channels.shape)

                df = pd.DataFrame({'DecayTaus': simulation_decay_taus, 'Amplitudes': simulation_amplitudes,
                                   'open_channels': open_channels.values})

                fallback_cmap = cm.get_cmap('viridis')
                df['open_channels'] = df['open_channels'].apply(
                    lambda x: float(ast.literal_eval(x)[0]) if isinstance(x, str) else float(x))

                norm = mcolors.Normalize(vmin=100, vmax=1500)

                sc = None

                for label, group in df.groupby('open_channels'):
                    # Use continuous colormap
                    sc = plt.scatter(group['DecayTaus'], group['Amplitudes'],
                                     c=[label] * len(group), cmap=fallback_cmap, norm=norm,
                                     label=f'{label}')

                # Add colorbar if any point used the continuous colormap
                if sc is not None:
                    cbar = plt.colorbar(sc)
                    cbar.set_label("Open Channels")
                # Add common plot elements
                plt.title(f"Decay Taus vs. Amplitude for {file[:-4]}")
                plt.xlabel("Decay Taus")
                plt.ylabel("Max Amplitudes (pA)")
                # plt.xlim(0, 0.4)
                # plt.ylim(0, 1000)

                # Save or show the completed plot
                plt.savefig(f"{file[:-4]}-n-decay.png")
                plt.show()

def Amplitude_Decay_Simulation_Colored_Glu():
    print("Beginning decay tau/amplitude analysis")
    folder_path = "C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\Scripts\\Experiments\\Simulation_Glu_N_Settings_Investigation\\EPSC_Data"
    cmap = cm.get_cmap('tab20', 20)  # 40 samples for smooth interpolation
    label_colors = {i: cmap(i / 19) for i in np.arange(0, 10, 0.5)}

    for file in os.listdir(folder_path):
        if file.endswith(".xlsx"):
            file_path = os.path.join(folder_path, file)
            print(f"Creating histogram for file {file_path}")
            simulation_EPSCs = pd.read_excel(file_path,sheet_name=0)
            channel_data = pd.read_excel(file_path,sheet_name="Channel Data")
            glut_data = channel_data['Glutamate Concentration']
            simulation_decay_taus = egg.tau_graph_generator(simulation_EPSCs, folder_name=folder_path,
                                                            plt_show=False,time=16)
            simulation_amplitudes = egg.amplitude_histogram_creator(simulation_EPSCs, folder_name=folder_path,
                                                                    plt_show=False)
            plt.close('all')
            print(len(simulation_amplitudes))
            print(len(simulation_decay_taus))
            print(glut_data.shape)

            df = pd.DataFrame({'DecayTaus':simulation_decay_taus,'Amplitudes':simulation_amplitudes,'Glu':channel_data['Glutamate Concentration'].values})

            fallback_cmap = cm.get_cmap('Blues')
            df['Glu'] = df['Glu'].apply(lambda x: float(ast.literal_eval(x)[0]) if isinstance(x, str) else float(x))

            norm = mcolors.Normalize(vmin=df['Glu'].min(), vmax=df['Glu'].max())

            sc = None

            for label, group in df.groupby('Glu'):
                if label in label_colors:
                    color = label_colors[label]
                    plt.scatter(group['DecayTaus'], group['Amplitudes'], color=color)
                else:
                    # Use continuous colormap
                    sc = plt.scatter(group['DecayTaus'], group['Amplitudes'],
                                     c=[label] * len(group), cmap=fallback_cmap, norm=norm,
                                     label=f'{label}')

            # Add colorbar if any point used the continuous colormap
            if sc is not None:
                cbar = plt.colorbar(sc)
                cbar.set_label("Glutamate Concentration (mM)")
            else:
                plt.legend(title="Glutamate Concentration (mM)")
            # Add common plot elements
            plt.title(f"Decay Tau vs. Amplitude for {file[:-4]}")
            plt.xlabel("Decay Taus")
            plt.ylabel("Max Amplitudes (pA)")
            # plt.xlim(0, 0.4)
            # plt.ylim(0, 1000)

            # Save or show the completed plot
            plt.savefig(f"{file[:-4]}-glu-decay.png")
            plt.show()

def naive_diffusion_sim():

    fixed_folder_path = "C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\Scripts\\Experiments\\Simulation_Glu_N_Settings_Investigation\\Diffusion_Testing\\test_epscs.xlsx"

    glutamate_params = pd.DataFrame(
        [{"gl_mean": 2.5, "gl_sd": 1, "distribution_type": "fixed_value", "fixed_value": 5,"diffusion":True}])

    channel_params = pd.DataFrame(
        [{"distribution_type": "normal", "channel_sd": 700, "channel_mean": 2200, "fixed_value": None}])


    EPSCs_df, all_total_channel_nums, channel_data_df = EPSC_Calc(num_EPSCs=200,channel_params=channel_params,glutamate_params=glutamate_params, output_file_path=fixed_folder_path)


def Rise_Times_Amps_Diffusion_Sim():
    print("Beginning Diffusion Sim rise time/amplitude analysis")
    folder_path = r"C:\Users\jawad\Downloads\Python-EPSC-NSFA-Pipeline\Scripts\Experiments\Simulation_Glu_N_Settings_Investigation\Diffusion_Testing"

    for file in os.listdir(folder_path):
        if not file.endswith(".xlsx"):
            continue

        file_path = os.path.join(folder_path, file)
        print(f"Processing file {file_path}")

        # Load data once
        simulation_EPSCs = pd.read_excel(file_path, sheet_name=0)
        channel_data = pd.read_excel(file_path, sheet_name="Channel Data")

        # Extract features
        simulation_rise_times = egg.rise_times_histogram_creator(simulation_EPSCs, folder_name=folder_path, plt_show=False)
        simulation_amplitudes = egg.amplitude_histogram_creator(simulation_EPSCs, folder_name=folder_path, plt_show=False)

        open_channels = channel_data['Open Channels']
        glut_data = channel_data['Glutamate Concentration']

        # Clean up dataframes
        df = pd.DataFrame({
            'RiseTimes': simulation_rise_times,
            'Amplitudes': simulation_amplitudes,
            'OpenChannels': open_channels.values,
            'Glu': glut_data.values
        })

        # Convert string values to floats if needed
        for col in ['OpenChannels', 'Glu']:
            df[col] = df[col].apply(lambda x: float(ast.literal_eval(x)[0]) if isinstance(x, str) else float(x))

        # Set up figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

        # ---- Subplot 1: Open Channels ----
        ax = axes[0]
        norm = mcolors.Normalize(vmin=100, vmax=1500)
        sc1 = ax.scatter(df['RiseTimes'], df['Amplitudes'], c=df['OpenChannels'],
                         cmap=cm.viridis, norm=norm)
        cbar1 = fig.colorbar(sc1, ax=ax)
        cbar1.set_label("Open Channels")
        ax.set_title("Rise Times vs Amplitude (Open Channels)")
        ax.set_xlabel("Rise Times (ms)")
        ax.set_ylabel("Max Amplitudes (pA)")
        ax.set_xlim(0, 0.4)
        ax.set_ylim(0, 1000)

        # ---- Subplot 2: Glutamate Concentration ----
        ax = axes[1]
        norm = mcolors.Normalize(vmin=df['Glu'].min(), vmax=df['Glu'].max())
        sc2 = ax.scatter(df['RiseTimes'], df['Amplitudes'], c=df['Glu'],
                         cmap=cm.viridis, norm=norm)
        cbar2 = fig.colorbar(sc2, ax=ax)
        cbar2.set_label("Glutamate Concentration (mM)")
        ax.set_title("Rise Times vs Amplitude (Glutamate)")
        ax.set_xlabel("Rise Times (ms)")
        ax.set_xlim(0, 0.4)
        ax.set_ylim(0, 1000)

        plt.suptitle(f"Rise Times vs. Amplitude for {file[:-5]}", fontsize=14)
        plt.tight_layout()
        plt.show()



def Rise_Times_Diffusion_Colored_Glu():
    print("Beginning rise time/amplitude analysis")
    folder_path = "C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\Scripts\\Experiments\\Simulation_Glu_N_Settings_Investigation\\Diffusion_Testing"
    cmap = cm.get_cmap('tab20', 20)  # 40 samples for smooth interpolation
    label_colors = {i: cmap(i / 19) for i in np.arange(0, 10, 0.5)}

    for file in os.listdir(folder_path):
        if file.endswith(".xlsx"):
            file_path = os.path.join(folder_path, file)
            print(f"Creating histogram for file {file_path}")
            simulation_EPSCs = pd.read_excel(file_path,sheet_name=0)
            channel_data = pd.read_excel(file_path,sheet_name="Channel Data")
            glut_data = channel_data['Glutamate Concentration']
            simulation_rise_times = egg.rise_times_histogram_creator(simulation_EPSCs, folder_name=folder_path,
                                                                     plt_show=False)
            simulation_amplitudes = egg.amplitude_histogram_creator(simulation_EPSCs, folder_name=folder_path,
                                                                    plt_show=False)
            plt.close('all')
            print(len(simulation_amplitudes))
            print(len(simulation_rise_times))
            print(glut_data.shape)

            df = pd.DataFrame({'RiseTimes':simulation_rise_times,'Amplitudes':simulation_amplitudes,'Glu':channel_data['Glutamate Concentration'].values})

            fallback_cmap = cm.get_cmap('viridis')
            df['Glu'] = df['Glu'].apply(lambda x: float(ast.literal_eval(x)[0]) if isinstance(x, str) else float(x))

            norm = mcolors.Normalize(vmin=df['Glu'].min(), vmax=df['Glu'].max())

            sc = None

            for label, group in df.groupby('Glu'):
                if label in label_colors:
                    color = label_colors[label]
                    plt.scatter(group['RiseTimes'], group['Amplitudes'], color=color)
                else:
                    # Use continuous colormap
                    sc = plt.scatter(group['RiseTimes'], group['Amplitudes'],
                                     c=[label] * len(group), cmap=fallback_cmap, norm=norm,
                                     label=f'{label}')

            # Add colorbar if any point used the continuous colormap
            if sc is not None:
                cbar = plt.colorbar(sc)
                cbar.set_label("Glutamate Concentration (mM)")
            else:
                plt.legend(title="Glutamate Concentration (mM)")
            # Add common plot elements
            plt.title(f"Rise Times vs. Amplitude for {file[:-4]}")
            plt.xlabel("Rise Times (ms)")
            plt.ylabel("Max Amplitudes (pA)")
            plt.xlim(0, 0.4)
            plt.ylim(0, 1000)

            # Save or show the completed plot
            plt.savefig(f"{file[:-4]}.png")







            # plt.show()

if __name__ == "__main__":
    #create_fixed_simulation()
    #create_normn_fixedglu_simulation()
    # create_normaln_lognglu_simulation()
    # create_fixedn_lognglu_simulation()
    # NSFA_Simulation_Settings()
    # Rise_Times_Simulation_Settings()
    # Rise_Time_Physiological()
    #RiseTime_Amplitude_Clustering()
    # RiseTime_Simulation_Clustering()
    # Phys_DBSCAN()

    ##Looking at the graph of Rise Time vs Amplitude for Log(Glu), FixedN, there are some interesting clustering things
    #We want to label it now by A. The Glutamate and B. by the #of channels present
    # Rise_Times_Simulation_Colored_Glu()

    #For this run, I added a parameter "continous" to the simulation glutamate parameters
    #I realized here that my LOGNORMAL was never hitting -- it was selecting NORMAL parameters with that mu/sd

    # create_normn_normglu_continuous_data()

    #Ran again to get the simulation for cont
    # Rise_Times_Simulation_Colored_Glu()


    #New identical function for running with amplitude coloring
    # Rise_Times_Simulation_Colored_N()


    #Next, we wanted to see what the drug_condition data looks like
    # RiseTime_Amplitude_Clustering_IEM()

    #A new optimization was done while glutamate was continuous. Running this again..
    # Rise_Times_Simulation_Colored_Glu()
    # Rise_Times_Simulation_Colored_N()

    #What does NSFA look like for our new optimization?
    # NSFA_continous_optimized()

    #If we run NSFA with ALL options on the optimization, what do we get?
    # NSFA_continous_optimized_all_matrix_options()

    #Some very strange looking things. Let's try a control.
    NSFA_control_all_options()

    #Are we 1000% sure the control is a control
    # create_fixed_simulation()

    #Figured out the issue. Now moving on to Amplitude VS Decay Plots
    # Amplitude_Decay_Simulation_Analysis()

    # Amplitude_Decay_Simulation_Colored_Glu()

    #Experimentally, let's try some naive models of diffusion.
    # naive_diffusion_sim()
    # Rise_Times_Amps_Diffusion_Sim()


