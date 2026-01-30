import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import Scripts.EPSC_Webapp.EPSC_App_Connection as EPSC_App_Connection
import Scripts.EPSC_Webapp.EPSC_preprocessing as EPSC_preprocessing
from tqdm import tqdm
import seaborn as sns
import itertools
import matplotlib.lines as mlines




params = {
    "alignment":["peak","midpoint","max_dv_dt"],
    "analysis_start_point":["peak_start","alignment_point"],
    "scaling":["minimize_error","peak_scaling_at_peak_time","peak_to_peak_scaling"],
    "output": ["linear","parabolic"],
    #"file_name":'C:/Users/jawad/Downloads/Python-EPSC-NSFA-Pipeline/Poster_Figures/Run 2 - Channel Changes, Fixed [GL]/EPSCs_Dataframe_normal_1000traces_10mM_1000channels.xlsx' ,
    "file_name": 'C:/Users/jawad/Downloads/Python-EPSC-NSFA-Pipeline/Poster_Figures/Run 1 - Rise times and Amplitude aligned/EPSCs_Dataframe_normal_1000traces_lognormalglut_1000channels.xlsx',
    "folder_name": "C:/Users/jawad/Downloads/Python-EPSC-NSFA-Pipeline/matrix_outputs/"
}


IEM_Params = {
    "alignment": ["peak"],
    "analysis_start_point": ["peak_start"],
    "scaling": ["minimize_error","peak_to_peak_scaling"],
    "output": ["linear", "parabolic"],
    "file_name": 'IEM_Data/IEM_EPSCs.xlsx',
    "folder_name": "C:/Users/jawad/Downloads/Python-EPSC-NSFA-Pipeline/matrix_outputs/IEM_Results/"

}

Control_Params = {
    "alignment": ["peak"],
    "analysis_start_point": ["peak_start"],
    "scaling": ["minimize_error","peak_to_peak_scaling"],
    "output": ["linear", "parabolic"],
    "file_name": "data-files/Control_Experimental_EPSCs_Juan.xlsx",
    "folder_name": "C:/Users/jawad/Downloads/Python-EPSC-NSFA-Pipeline/matrix_outputs/Control_Results/"

}

CNQX_Kyn_Params = {
    "alignment": ["peak"],
    "analysis_start_point": ["peak_start"],
    "scaling": ["minimize_error","peak_to_peak_scaling"],
    "output": ["linear", "parabolic"],
    "file_name": 'C:/Users/jawad/Downloads/Python-EPSC-NSFA-Pipeline/data-files/CNQX_Kyn_EPSCs.xlsx',
    "folder_name": "C:/Users/jawad/Downloads/Python-EPSC-NSFA-Pipeline/matrix_outputs/CNQX_Results/"

}

uniform_channel_params = {
    "alignment": ["peak"],
    "analysis_start_point": ["peak_start"],
    "scaling": ["minimize_error","peak_to_peak_scaling"],
    "output": ["linear", "parabolic"],
    "file_name": "C:/Users/jawad/Downloads/Python-EPSC-NSFA-Pipeline/EPSCs-uniformchannels.pkl",
    "folder_name": "C:/Users/jawad/Downloads/Python-EPSC-NSFA-Pipeline/"
}
def matrix_generator(params,first_sheet=True,debug=False,save_figs=True): #Where params is a dictionary of parameters
    matrix = []
    if not params["direct_df_input"]:
        if params["file_name"].split('.')[-1] == 'xlsx':
            excel_file = pd.ExcelFile(f"{params['file_name']}")
        if first_sheet:
            sheet_names = [excel_file.sheet_names[0]]
        else:
            sheet_names = excel_file.sheet_names
    else:
        sheet_names = params["sheet_names"]
    workflow_report = {'files processed':sheet_names}
    for sheet_name in tqdm(sheet_names, desc="Processing Excel files", unit="file"):
        workflow_report[sheet_name] = {}
        if not params["direct_df_input"]:
            epscs = pd.read_excel(params["file_name"],sheet_name=sheet_name)
        else:
            epscs = params["EPSCs"]
        epscs = epscs.to_numpy()
        folder_name = params["folder_name"]
        if params["recording_duration"] == None:
            time_duration = 16  #ms
        else:
            time_duration = params["recording_duration"]
        num_samples = epscs.shape[0]
        print(f"Shape of EPSCs: {epscs.shape}")
        ##Plotting the raw EPSCs, unchanged

        if save_figs:
            fig, axs = plt.subplots(1, 1)
            timepoints = np.linspace(0, time_duration, num_samples)
            for i in range(epscs.shape[1]):
                axs.plot(timepoints, epscs[:, i], label=f'Trace {i + 1}')
            axs.set_xlabel('Time (ms)', fontsize=13)
            axs.set_ylabel('Current (pA)', fontsize=13)
            axs.set_title('EPSCs Before Processing')
            plt.savefig(f'{folder_name}Raw_EPSCs_{sheet_name}.png')
            workflow_report[sheet_name]['raw_epsc_fig_path'] = f'{folder_name}Raw_EPSCs_{sheet_name}.png'


        ##Implementing alignment options
        #for each alignment option:
        for alignment_type in params["alignment"]:
            workflow_report[sheet_name][alignment_type] = {}
            if alignment_type == "peak":
                alignment_processed, peak_index = EPSC_preprocessing.align_peaks(epscs)
                start_point = peak_index


            elif alignment_type == "midpoint":
                alignment_processed = EPSC_preprocessing.align_midpoint(epscs)
                start_point = 15
            elif alignment_type == "max_dv_dt":
                alignment_processed = EPSC_preprocessing.align_dv_dt(epscs,debug=debug)
                start_point = 15
            elif alignment_type == "None":
                alignment_processed = epscs

            if save_figs:
                fig, axs = plt.subplots(1, 2)
                timepoints = np.linspace(0, time_duration, num_samples)
                for i in range(epscs.shape[1]):
                    axs[0].plot(timepoints, epscs[:, i], label=f'Trace {i + 1}')
                    axs[1].plot(timepoints, alignment_processed[:, i], label=f'Trace {i + 1}')
                axs[0].axvline(x=15 * (time_duration/num_samples), color='red', linestyle='--', linewidth=2, label='Alignment Point')
                axs[0].set_xlabel('Time (ms)', fontsize=13)
                axs[0].set_ylabel('Current (pA)', fontsize=13)
                axs[0].set_title('EPSCs Before Alignment')
                axs[1].axvline(x=15 *  (time_duration/num_samples), color='red', linestyle='--', linewidth=2, label='Alignment Point')
                axs[1].set_xlabel('Time (ms)', fontsize=13)
                axs[1].set_ylabel('Current (pA)', fontsize=13)
                axs[1].set_title(f'EPSCs After Alignment: {alignment_type}')
                plt.savefig(f'{folder_name}{alignment_type}_Alignment.png')
            if debug:
                plt.show()
                workflow_report[sheet_name][alignment_type]['image'] = f'{folder_name}{alignment_type}_Alignment.png'
            #Create the template
            try:
                timepoints, template = EPSC_App_Connection.create_template(alignment_processed,time_duration,num_samples)
            except Exception as e:
                print("Empty trace!")
                continue
            # workflow_report[sheet_name][alignment_type]['template_data'] = template
            workflow_report[sheet_name][alignment_type]['template_image'] = f'{folder_name}{alignment_type}_Alignment.png'
            template_max = np.max(template)
            peak_index = np.argmax(template)
            # print(f"Template max: {template_max} and {template[peak_index]}")

            #Set some variables for the analysis
            pool_indices = EPSC_App_Connection.create_pool_indices(alignment_processed, peak_index)
            num_traces = alignment_processed.shape[1]
            raw_sorted = EPSC_App_Connection.sort_EPSCs_by_size(alignment_processed, peak_index)
            sampling_rate = num_samples/time_duration #yields sample/ms
            four_seconds_point = sampling_rate * 4 #4ms * xsamples/ms
            endPoint = int(four_seconds_point) #template.shape[0] - 1
            print(f"Endpoint: {endPoint}")

            #For each scaling option
            for scaling_type in params["scaling"]:
                if scaling_type == "minimize_error":
                    residuals_array = EPSC_App_Connection.create_residuals(num_traces,raw_sorted,template,
                                                                                                  error_minimize=True,debug=debug)

                elif scaling_type == "peak_scaling_at_peak_time":
                    residuals_array = EPSC_App_Connection.create_residuals(num_traces, raw_sorted,
                                                                                                  template,
                                                                                                  error_minimize=False,peak_to_peak=False,debug=debug)


                elif scaling_type == "peak_to_peak_scaling":
                    residuals_array = EPSC_App_Connection.create_residuals(num_traces, raw_sorted,template,

                                                                                                    error_minimize=False,peak_to_peak=True,debug=debug)
                elif scaling_type == "raw":
                    residuals_array = EPSC_App_Connection.raw_residuals(raw_sorted,template)
                #For each analysis start point option
                for start_option in params["analysis_start_point"]:
                    segment_indices = EPSC_App_Connection.create_segment_indices(template, start_option,endPoint)


                    if debug:
                        plt.figure()
                        plt.plot(timepoints, template, color='blue')
                        plt.title("Segment Validation on Template")
                        plt.xlabel("Time (ms)")
                        plt.ylabel("Current (pA)")
                        for index in segment_indices:
                            plt.axvline(x=(peak_index + index) * (time_duration/num_samples), color='red', linestyle='--', linewidth=1)
                        plt.savefig("Segments_Validation.png")
                        workflow_report[sheet_name][start_option] = 'Segments_Validation.png'
                        plt.show()
                    # if start_option == "peak_start":
                    #Run mean variance with that start point
                    means = EPSC_App_Connection.mean_calculation(raw_sorted, start_index=peak_index, endPoint=endPoint, segment_indices=segment_indices,
                                             analysis_type=3)
                    vars = EPSC_App_Connection.var_calculation(peak_index, residuals_array, segment_indices, pool_indices, endPoint, 3)


                    # elif start_option == "alignment_point":
                    #
                    #      means = EPSC_App_Connection.mean_calculation(raw_sorted, start_index=start_point, endPoint=endPoint, segment_indices=segment_indices,
                    #                               analysis_type=3)
                    #      vars = EPSC_App_Connection.var_calculation(start_point, residuals_array, segment_indices, pool_indices, endPoint, 3)
                    #
                    #     #Run mean variance with that alignment

                    #Derive i, n values for linear and parabolic
                    try:
                        fit_parabola, roots, initial_slope = EPSC_App_Connection.fitting_parabola(means, vars,force_linear=False)
                        n = roots[0] / initial_slope

                    except Exception as e:
                        print(f"⚠️ Skipping sweep due to error in fitting_parabola: {e}")
                        continue
                    lin_fit_parabola, lin_roots, lin_initial_slope = EPSC_App_Connection.fitting_parabola(means, vars,force_linear=True)
                    matrix_entry = {"cell_name": sheet_name,"alignment":alignment_type,"analysis_start_point":start_option,"scaling":scaling_type,"linear_i":lin_initial_slope,"parabolic_i":initial_slope,"num_channels":n,"template_max":template_max}
                    matrix.append(matrix_entry)
                    if save_figs:
                        print("Saving figs...")
                        fig, axs = plt.subplots(1, 1)
                        axs.scatter(means, vars, color='green')
                        for idx,item in enumerate(means):
                            axs.text(means[idx],vars[idx],idx)
                        sorter = np.sort(means)
                        roots = fit_parabola.r
                        if len(roots) > 1:
                            x_vals = np.linspace(min(roots) - 1, max(roots) + 1, 500)
                            axs.plot(x_vals, fit_parabola(x_vals), color='black')

                        else:
                            axs.plot(sorter,lin_fit_parabola(sorter),color='red')
                        if "mean_channels" in params:
                            #Plot the idealized parabola
                            N = params["mean_channels"]
                            i = params["theoretical_current"]
                            ideal_parabola = np.poly1d([-1/N,i,0])
                            roots = ideal_parabola.r
                            x_vals = np.linspace(min(roots) - 1, max(roots) + 1, 500)
                            axs.plot(x_vals,ideal_parabola(x_vals),color='blue')
                        axs.plot(sorter[0:20], lin_fit_parabola(sorter)[0:20], color='red')
                        axs.set_title("Variance vs Mean")
                        axs.set_xlabel("Mean Current (pA)")
                        axs.set_ylabel("Current variance (pA^2)")

                        #plt.savefig(f"{params['folder_name']}/{alignment_type}_{start_option}_{scaling_type}_{sheet_name}.png")
                        plt.show()
                    print(matrix_entry)
    return matrix



if __name__ == "__main__":
    #
    # matrix = matrix_generator(IEM_Params)
    # matrix_df = pd.DataFrame(matrix)
    # matrix_df.to_pickle(f"{IEM_Params['folder_name']}IEM_Matrix_tempmax.pkl")
    # matrix_df.to_excel(f"{IEM_Params['folder_name']}IEM_Matrix_tempmax.xlsx")
    # #
    # matrix = matrix_generator(Control_Params)
    # matrix_df = pd.DataFrame(matrix)
    # matrix_df.to_pickle(f"{Control_Params['folder_name']}Control_Matrix_tempmax.pkl")
    # matrix_df.to_excel(f"{Control_Params['folder_name']}Control_Matrix_tempmax.xlsx")
    #
    # matrix = matrix_generator(CNQX_Kyn_Params)
    # matrix_df = pd.DataFrame(matrix)
    # matrix_df.to_pickle(f"{CNQX_Kyn_Params['folder_name']}CNQX_Matrix_tempmax.pkl")
    # matrix_df.to_excel(f"{CNQX_Kyn_Params['folder_name']}CNQX_Matrix_tempmax.xlsx")


    matrix = matrix_generator(uniform_channel_params)
    matrix_df = pd.DataFrame(matrix)
    matrix_df.to_excel(f"Uniform_Channel_Log_Glut.xlsx")

    CNQX_Kyn_matrix = pd.read_excel(f"{CNQX_Kyn_Params['folder_name']}CNQX_Matrix.xlsx")
    IEM_matrix = pd.read_excel(f"{IEM_Params['folder_name']}IEM_Matrix.xlsx")
    Control_matrix = pd.read_pickle(f"{Control_Params['folder_name']}Control_Matrix.pkl")

    print(IEM_matrix)

    CNQX_Kyn_matrix['Dataset'] = 'CNQX_Kyn'
    IEM_matrix['Dataset'] = 'IEM'
    Control_matrix['Dataset'] = 'Control'
    full_matrix = pd.concat([CNQX_Kyn_matrix,IEM_matrix,Control_matrix])

    error_min_dataset = full_matrix[full_matrix['scaling'] == 'minimize_error']
    # peak_scaling_dataset = full_matrix[full_matrix['scaling'] == 'peak_to_peak_scaling']


    error_min_dataset['base_cell'] = error_min_dataset['cell_name'].apply(lambda x: x.split('_')[0])
    colors = ['#7f7f7f', '#bcbd22', '#17becf','#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    unique_cells = error_min_dataset['base_cell'].unique()
    cell_colors = {cell: colors[i % len(colors)] for i, cell in enumerate(unique_cells)}

    # Define marker styles
    marker_styles = ['o', 's', 'D', '^', 'v', 'p', '*', 'X', 'P', '<', '>']
    cell_markers = {cell: marker for cell, marker in zip(unique_cells, itertools.cycle(marker_styles))}

    legend_handles = [mlines.Line2D([], [], color=color, marker=cell_markers[cell], linestyle='None', markersize=8, label=cell) for cell, color in cell_colors.items()]

    legend_fig, legend_ax = plt.subplots(figsize=(2, len(legend_handles) * 0.5))  # Adjust figure size based on the number of cells
    legend_ax.axis("off")  # Hide axes

    # Create a legend in a standalone figure
    legend_ax.legend(handles=legend_handles, title="Control Cells", loc="center", frameon=False)
    plt.show()





    ####Plotting Individual points, change by drug

    ###IEM versus Control
    # Prepare your data
    iem = error_min_dataset[error_min_dataset['Dataset'] == "IEM"][['base_cell','linear_i','num_channels','[IEM]','IEM_Type']]
    control = error_min_dataset[error_min_dataset['Dataset'] == "Control"][['base_cell','linear_i','num_channels']]

    iem_control_merged = iem.merge(control, on='base_cell', suffixes=('_IEM','_Control'))

    # Define color palette

    # # Map unique base_cells to colors
    # unique_cells = iem_control_merged['base_cell'].unique()
    # cell_colors = {cell: colors[i % len(colors)] for i, cell in enumerate(unique_cells)}

    # # Define marker styles
    # marker_styles = ['o', 's', 'D', '^', 'v', 'p', '*', 'X', 'P', '<', '>']
    # cell_markers = {cell: marker for cell, marker in zip(unique_cells, itertools.cycle(marker_styles))}

    # Create figure
    fig, ax = plt.subplots(figsize=(7, 5))

    # Create custom handles for legend
    solid_line = mlines.Line2D([], [], color='black', linestyle='-', label='IEM1460')
    dashed_line = mlines.Line2D([], [], color='black', linestyle='--', label='IEM1925')

    # Plot lines and points
    for i, row in iem_control_merged.iterrows():
        marker = cell_markers[row['base_cell']]
        color = cell_colors[row['base_cell']]

        # Adjust line thickness based on [IEM]
        weight = row['[IEM]'] / max(iem_control_merged['[IEM]']) * 5  # Scale weight

        # Set line style based on IEM_Type
        if row['IEM_Type'] == 'IEM1460':
            line_style = '--'
        else:
            line_style = '-'

        # Connect points with the selected line style
        plt.plot([1, 2], [row['linear_i_Control'], row['linear_i_IEM']], color=color, alpha=0.7, linewidth=weight, linestyle=line_style)

        # Scatter points
        plt.scatter(1, row['linear_i_Control'], color=color, marker=marker)
        plt.scatter(2, row['linear_i_IEM'], color=color, marker=marker)

    # Formatting
    plt.xticks([1, 2], ['Control', 'IEM'])
    plt.xlabel("Cell Treatment Condition")
    plt.ylabel("Estimated Unitary Current (pA)")
    plt.title("Unitary Current versus Cell Condition")

    # Add the legend
    plt.legend(handles=[solid_line, dashed_line])

    # Save and show
    plt.savefig('individual_trial_plots/iem_vs_control_i.png')
    plt.show()


    ##Num channels
    plt.figure(figsize=(6,4))

    for i, row in iem_control_merged.iterrows():
        # Connect points
        marker = cell_markers[row['base_cell']]
        color = cell_colors[row['base_cell']]
        weight = row['[IEM]'] / max(iem_control_merged['[IEM]']) * 5  # Scale weight

        # Set line style based on IEM_Type
        if row['IEM_Type'] == 'IEM1460':
            line_style = '--'
        else:
            line_style = '-'
        plt.plot([1, 2], [row['num_channels_Control'], row['num_channels_IEM']],color=color, alpha=0.5,linestyle=line_style,linewidth=weight)

        # Scatter points with different markers
        plt.scatter(1, row['num_channels_Control'], color=color, marker=marker, label='Control' if i == 0 else "")
        plt.scatter(2, row['num_channels_IEM'], color=color, marker=marker, label='IEM' if i == 0 else "")


    # Formatting
    plt.xticks([1, 2], ['Control', 'IEM'])
    plt.xlabel("Cell Treatment Condition")
    plt.legend(handles=[solid_line, dashed_line])
    plt.ylabel("Estimated Number of Channels")
    plt.title("Number of Available Channels vs Cell Condition")
    plt.ylim(0, 800) # Y-axis from -0.5 to 1

    plt.savefig('individual_trial_plots/iem_vs_control_n.png')

    plt.show()


    ###CNQX/KYN versus control

    solid_line = mlines.Line2D([], [], color='black', linestyle='-', label='CNQX')
    dashed_line = mlines.Line2D([], [], color='black', linestyle='--', label='KYN')
    cnqx = error_min_dataset[error_min_dataset['Dataset']== "CNQX_Kyn"][['base_cell','linear_i','num_channels','Drug_Type']]

    cnqx_control_merged = cnqx.merge(control,on='base_cell',suffixes=('_CNQX','_Control'))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']


    # Create a mapping from 'base_cell' to a unique color
    unique_cells = cnqx_control_merged['base_cell'].unique()
    cell_colors = {cell: colors[i % len(colors)] for i, cell in enumerate(unique_cells)}

    plt.figure(figsize=(6,4))

    for _, row in cnqx_control_merged.iterrows():
        # Connect points
        marker = cell_markers[row['base_cell']]
        color = cell_colors[row['base_cell']]
        if row['Drug_Type'] == 'CNQX':
            line_style = '-'
        else:
            line_style = '--'
        plt.plot([1, 2], [row['linear_i_Control'], row['linear_i_CNQX']], color=color, alpha=0.5,linestyle=line_style)


        # Scatter points with different markers
        plt.scatter(1, row['linear_i_Control'], color=color, marker=marker, label='Control' if _ == 0 else "")
        plt.scatter(2, row['linear_i_CNQX'], color=color, marker=marker, label='CNQX' if _ == 0 else "")


    # Formatting
    plt.xticks([1, 2], ['Control', 'CNQX/Kyn'])
    plt.xlabel("Cell Treatment Condition")
    plt.ylabel("Estimated Unitary Current (pA)")
    plt.title("Unitary Current versus Cell Condition")
    plt.ylim(.6, 1.8) # Y-axis from -0.5 to 1
    plt.legend(handles=[solid_line, dashed_line])
    plt.savefig('individual_trial_plots/cnqx_vs_control_i.png')

    plt.show()

    ##Num channels
    plt.figure(figsize=(6,4))

    for _, row in cnqx_control_merged.iterrows():
        # Connect points
        marker = cell_markers[row['base_cell']]
        current_cell = row['base_cell']
        color = cell_colors[row['base_cell']]

        if row['Drug_Type'] == 'CNQX':
            line_style = '-'
        else:
            line_style = '--'
        plt.plot([1, 2], [row['num_channels_Control'], row['num_channels_CNQX']],color=color, alpha=0.5,linestyle=line_style)

        # Scatter points with different markers
        plt.scatter(1, row['num_channels_Control'], color=color, marker=marker, label='Control' if _ == 0 else "")
        plt.scatter(2, row['num_channels_CNQX'], color=color, marker=marker, label='IEM' if _ == 0 else "")
    #Limit from .6 to 1.8 for current y axis
    #Limit from 100 to 700 for channels y axis

    # Formatting
    plt.xticks([1, 2], ['Control', 'CNQX/Kyn'])
    plt.xlabel("Cell Treatment Condition")
    plt.legend(handles=[solid_line, dashed_line])
    plt.ylabel("Estimated Number of Channels")
    plt.title("Number of Available Channels vs Cell Condition")
    plt.ylim(0, 800) # Y-axis from -0.5 to 1

    plt.savefig('individual_trial_plots/cnqx_vs_control_n.png')

    plt.show()









    ###Error Minimization Figures -- MEANS
    # fig,axes = plt.subplots(1,2,figsize=(12,5))
    # sns.boxplot(data=error_min_dataset,x='Dataset',y='linear_i',ax=axes[0])
    # axes[0].set_title("Linear-Extracted Unitary Current by Trial")
    #
    # sns.boxplot(data=error_min_dataset,x='Dataset',y='parabolic_i',ax=axes[1])
    # axes[1].set_title("Parabolic-Extracted Unitary Current by Trial")
    # plt.savefig("i-boxplots-errormin.png", dpi=300, bbox_inches="tight")
    # plt.show()
    #
    #
    # fig2, axes2 = plt.subplots(1,2,figsize=(12,5))
    # sns.boxplot(data=error_min_dataset,x='Dataset',y='num_channels',ax=axes2[0])
    # axes2[0].set_title("Number of Channels by Trial - Error Minimization")
    #
    # sns.boxplot(data=peak_scaling_dataset,x='Dataset',y='num_channels',ax=axes2[1])
    # axes2[1].set_title("Number of Channels by Trial - Peak Scaling")
    # plt.savefig("n-boxplots.png", dpi=300, bbox_inches="tight")
    # plt.show()
    #
    # fig3,axes3 = plt.subplots(1,2,figsize=(12,5))
    # sns.boxplot(data=peak_scaling_dataset,x='Dataset',y='linear_i',ax=axes3[0])
    # axes3[0].set_title("Linear-Extracted Unitary Current by Trial")
    # sns.boxplot(data=peak_scaling_dataset,x='Dataset',y='parabolic_i',ax=axes3[1])
    # axes3[1].set_title("Parabolic-Extracted Unitary Current by Trial")
    # plt.savefig("i-boxplots-peakscaling.png", dpi=300, bbox_inches="tight")
    # plt.show()




    # print(full_matrix.head())