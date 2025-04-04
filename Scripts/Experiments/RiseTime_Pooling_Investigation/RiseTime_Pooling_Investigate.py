from Scripts.EPSC_Simulation.EPSC_Simulation import EPSC_Calc
import pandas as pd
import matplotlib.pyplot as plt
from Scripts.NSFA_Tools.EPSC_Matrix_Generator import matrix_generator
from Scripts.EPSC_Webapp.EPSC_App_Connection import multi_pool_analysis
import Scripts.NSFA_Tools.EPSC_Graphic_Utilities as egg

''' In a previous run, it was shown that simulations created with a lognormal glutamate distribution essentially "break"
the NSFA. Our hypothesis is that this is because of the variance created in the rise time. The goal of this script is to
 separate simulated EPSCs that match the CDF of physiological ones into pools by rise time, then run NSFA and see
 if it will retrieve expected EPSC results'''

########SECTION 1: Simulation Creation and Validation########

#First we want to create a new run of the lognormal simulation that will have boundaries of 1 and 10mM glutamate

#Bounds: Channels - [50,4000] Glutamate - [1,100]
def create_control_EPSCs():
    glutamate_params = pd.DataFrame([{"gl_mean": 3.5, "gl_sd": 1, "distribution_type": "lognormal","fixed_value":None}])

    channel_params = pd.DataFrame([{"distribution_type":"normal", "channel_sd":560,"channel_mean":1800,"fixed_value":None}])

    EPSCs_df,all_total_channel_nums,channel_data_df= EPSC_Calc(num_EPSCs=1000,channel_params=channel_params,glutamate_params=glutamate_params)

    EPSCs_df.to_pickle('EPSCs_for_risetime_invest_100bound_debug.pkl')
    channel_data_df.to_pickle('glutinfo_for_risetime_invest_100bound_debug.pkl')

#We also want to record the glutamate distribution being used
def visualize_Glu():
    channel_data = pd.read_excel('EPSCs_for_rise_time.xlsx',sheet_name='Channel Data')
    # EPSCs = pd.read_excel('EPSCs_for_rise_time.xlsx',sheet_name = 'EPSCs')
    print(channel_data["Glutamate Concentration"])
    #Then we'll visualize the glutamate distribution
    plt.figure(figsize=(8, 5))  # Set figure size
    plt.hist(
        channel_data["Glutamate Concentration"],
        bins=20,  # Specify the number of bins
        color='skyblue',  # Bar color
        edgecolor='black',  # Add edges to bars
        alpha=0.7  # Add transparency
    )
    plt.title("Distribution of Glutamate Concentration", fontsize=14)  # Add a title
    plt.xlabel("Glutamate Concentration (mM)", fontsize=12)  # Label x-axis
    plt.ylabel("Frequency", fontsize=12)  # Label y-axis
    plt.tight_layout()  # Adjust spacing to fit labels
    plt.savefig("Glutamate_Hist_Control.png")

    plt.show()


#Then, we want to check that the CDFs are valid
#This was done in the simulation-optimization folder, but TODO: create a utility for CDF validation

#########SECTION 2: NSFA CONTROL - Problem Validation########

#Next, we want to run the full matrix of NSFA on the EPSCs. We can create a heatmap once again showing methodical error
#We should also generate the graphs for the NSFA runs so that we can use them in a side-by-side example

def NSFA_problem_validation():
    # params = {
    #     "alignment": ["peak", "midpoint", "max_dv_dt"],
    #     "analysis_start_point": ["peak_start", "alignment_point"],
    #     "scaling": ["minimize_error", "peak_scaling_at_peak_time", "peak_to_peak_scaling"],
    #     "output": ["linear", "parabolic"],
    #     "file_name": 'EPSCs_for_rise_time.xlsx',
    #     "folder_name": "C:/Users/jawad/Downloads/Python-EPSC-NSFA-Pipeline/Scripts/Experiments/RiseTime_Pooling_Investigation"
    #
    # }

    params = {
        "alignment": ["peak"],
        "analysis_start_point": ["peak_start"],
        "scaling": ["minimize_error", "peak_scaling_at_peak_time", "peak_to_peak_scaling"],
        "output": ["linear", "parabolic"],
        "file_name": 'EPSCs_for_rise_time.xlsx',
        "folder_name": "C:/Users/jawad/Downloads/Python-EPSC-NSFA-Pipeline/Scripts/Experiments/RiseTime_Pooling_Investigation"

    }

    matrix = matrix_generator(params)
    matrix_df = pd.DataFrame(matrix)
    matrix_df.to_pickle("NSFA_Control_Matrix.pkl")
    matrix_df.to_excel("NSFA_Control_Matrix.xlsx")



#########SECTION 3: Assessing Rise Time Distribution########
#Here, we want to first graph out the simulation rise times, choose a bin/pool size based on that range
#This was done in the CDF call earlier

#########SECTION 4: NSFA Binned by Rise Times - New Method Test ########
#In this final section, we want to run the NSFA in such a way that the pools are being created for rise-times
#We will need to edit the NSFA analysis function to include that functionality.
#Finally, we will need to run the same heatmap to show the error across i for the new pooled method



#TODO: Multipool analysis needs the same functionality as one pool analysis so that we can run multiple types of data through
def NSFA_pooled_risetimes():

    #Sort EPSCs by rise time
    EPSCs = pd.read_excel('EPSCs_for_rise_time.xlsx')
    #egg.rise_times_histogram_creator(EPSCs,folder_name="Scripts/") #Ran this then deleted the "Trace" column
    rise_times = pd.read_excel('rise_times.xlsx')
    bins = 4
    rise_times['bin'] = pd.cut(
        rise_times.iloc[:, 0], bins=bins, labels=[f"Bin {i + 1}" for i in range(bins)], include_lowest=True
    )

    print(f"Sorted Rise Times: {rise_times.iloc[:,1]}")
    rise_times.to_excel("risetimedebug.xlsx")
    # Step 2: Group column indices of the (500,500) dataframe based on bins
    # Use .iloc to reference the positional indices of columns
    bin_groups = {}
    for bin_label, group in rise_times.groupby('bin'):
        bin_groups[bin_label] = group.index.tolist()
        print(group)

    # Step 3: Separate columns of the (500,500) dataframe into groups
    output_file = "grouped_columns.xlsx"
    print(bin_groups)
    with pd.ExcelWriter(output_file) as writer:
        for bin_label, column_indices in bin_groups.items():
            # Use iloc to select columns by their positional indices
            grouped_columns = EPSCs.iloc[:, column_indices]

            # Write the group to an Excel sheet
            grouped_columns.to_excel(writer, sheet_name=str(bin_label),index=False,header=False)

    print(f"Data has been saved to {output_file}.")
    # params = {
    #     "alignment": ["peak", "midpoint", "max_dv_dt"],
    #     "analysis_start_point": ["peak_start", "alignment_point"],
    #     "scaling": ["minimize_error", "peak_scaling_at_peak_time", "peak_to_peak_scaling"],
    #     "output": ["linear", "parabolic"],
    #     "file_name": 'grouped_columns.xlsx',
    #     "folder_name": "C:/Users/jawad/Downloads/Python-EPSC-NSFA-Pipeline/Scripts/Experiments/RiseTime_Pooling_Investigation"
    #
    # }
    params = {
        "alignment": ["peak"],
        "analysis_start_point": ["peak_start"],
        "scaling": ["minimize_error", "peak_scaling_at_peak_time", "peak_to_peak_scaling"],
        "output": ["linear", "parabolic"],
        "file_name": 'grouped_columns.xlsx',
        "folder_name": "C:/Users/jawad/Downloads/Python-EPSC-NSFA-Pipeline/Scripts/Experiments/RiseTime_Pooling_Investigation"

    }
    experimental_matrix = matrix_generator(params,first_sheet=False)
    matrix_df = pd.DataFrame(experimental_matrix)
    matrix_df.to_excel("Experimental_Matrix_peaksonly.xlsx")






if __name__ == "__main__":
    # create_control_EPSCs()
    # visualize_Glu()
    #NSFA_problem_validation()
    NSFA_pooled_risetimes()













'''NOTES:
Upon running the EPSCs with the lognormal glutamate, it looks like we were really only getting 3 values of [Glu] anyway...'''