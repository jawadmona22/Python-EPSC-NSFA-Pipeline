from Scripts.NSFA_Tools.EPSC_Matrix_Generator import matrix_generator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

from Scripts.EPSC_Simulation.EPSC_Simulation import EPSC_Calc


#Question: Is there a systematic error in the glutamate/n relationship for a method that fails?

#Method of Investigation: Creating a heatmap to display the error for a 5x5 trial of n versus glutamate




#First: need to run and create the raw EPSCs

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


if __name__ == "__main__":

    # folder_name = 'Glutamate_Channel_Investigations_3-21'
    # directory = os.fsencode(folder_name)
    # all_matrices = []
    #
    # #First, we run NSFA on as many versions for the 25 simulated EPSC conditions
    # for file in os.listdir(directory):
    #     filename = os.fsdecode(file)
    #     if filename.endswith(".xlsx"):
    #         params = {
    #         "alignment": ["peak"],
    #         "analysis_start_point": ["peak_start"],
    #         "scaling": ["minimize_error","peak_to_peak_scaling"],
    #         "output": ["linear", "parabolic"],
    #         "file_name": f'{folder_name}/{filename}',
    #         "folder_name": f"C:/Users/jawad/Downloads/Python-EPSC-NSFA-Pipeline/{folder_name}/matrix_outputs"
    #         }
    #         single_matrix = pd.DataFrame(matrix_generator(params))
    #         single_matrix['parabolic_error'] = single_matrix['parabolic_i'] - .56
    #         single_matrix['lin_error'] = single_matrix['linear_i'] - .56
    #         print(type(single_matrix))
    #         stripped_name = filename.split('.')[0]
    #         match = re.search(r"_(\d+)ch_(\d+(\.\d+)?)mM", filename)
    #         if match:
    #             n_value = int(match.group(1))
    #             glu_value = float(match.group(2))  # Extract the value before 'mM' as a float
    #             single_matrix['n'] = n_value
    #             single_matrix['glu'] = glu_value
    #         single_matrix.to_pickle(f"{params['folder_name']}/{stripped_name}-matrix.pkl")
    #         all_matrices.append(single_matrix)
    #
    # #Then we set up the data to be the 2D columns we want: n and error from i.
    # #In this case, i = .56mV
    # combined_df = pd.concat(all_matrices, ignore_index=True)
    # combined_df.to_pickle("combined_dataframe.pkl")

    combined_df = pd.read_pickle('combined_dataframe.pkl')
    #For every unique type
    for alignment in combined_df["alignment"].unique():
        for analysis_start in combined_df["analysis_start_point"].unique():
            for scaling in combined_df["scaling"].unique():
                print(combined_df.columns)
                typed_dataframe = combined_df[
                    (combined_df['alignment'] == alignment) &
                    (combined_df['analysis_start_point'] == analysis_start) &
                    (combined_df['scaling'] == scaling)]
                linear_heatmap_data = typed_dataframe[['n','lin_error','glu']]
                parab_heatmap_data = typed_dataframe[['n', 'parabolic_error', 'glu']]

                # Pivot the dataframe to create a grid for the heatmap
                parab_pivot = parab_heatmap_data.pivot(index='n', columns='glu', values='parabolic_error')
                linear_pivot = linear_heatmap_data.pivot(index='n', columns='glu', values='lin_error')
                vmin = -.5
                vmax = .5
                # Create the heatmap
                fig, axes = plt.subplots(1, 2, figsize=(16, 8))  # 1 row, 2 columns
                sns.heatmap(parab_pivot, ax=axes[0], annot=True, fmt=".2f", cmap="RdBu", cbar_kws={'label': 'Value'},
                            vmin=vmin, vmax=vmax)

                # Add labels and title
                axes[0].set_title(f'Scaling: {scaling} Analysis_start:{analysis_start} Alignment {alignment}')
                axes[0].set_xlabel('Glu')
                axes[0].set_ylabel('n')

                sns.heatmap(linear_pivot,ax=axes[1] ,annot=True, fmt=".2f", cmap="RdBu",vmin=vmin, vmax=vmax)

                # Add labels and title
                axes[1].set_title(f' Scaling: {scaling} Analysis_start:{analysis_start} Alignment {alignment}')
                axes[1].set_xlabel('Glu')
                axes[1].set_ylabel('n')
                plt.tight_layout()
                # Show the plot
                # plt.show()

                plt.savefig(f'Scaling_{scaling}_Analysis_start_{analysis_start}_Alignment_{alignment}.png')

                #For linear

        #For parabolic









