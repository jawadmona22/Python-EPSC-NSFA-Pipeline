from Scripts.EPSC_Simulation.EPSC_Simulation import EPSC_Calc, Agonist_Pulse
import pandas as pd
from Scripts.NSFA_Tools.EPSC_Matrix_Generator import matrix_generator
import Scripts.NSFA_Tools.EPSC_Graphic_Utilities as egg
import numpy as np
import Scripts.EPSC_Webapp.EPSC_App_Connection as EPSC_App_Connection
import Scripts.EPSC_Webapp.EPSC_preprocessing as EPSC_preprocessing
from pandas import json_normalize
import seaborn as sns
import os
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import defaultdict
from openpyxl import load_workbook

'''
The purpose of this script is to investigate how changing the CP\\CI ratio and the current ratios between the two
changes the outcome of NSFA. 

The type of NSFA used will be always be PEAK START-PEAK ALIGNED-MINIMIZE ERROR

'''
# os.chdir('C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\Scripts\\Experiments\\CP_CI_Ratio_Investigation')


##STEP 1: Create 4 simulations matching 3x\\6x multiplier of CP\\CI current and 10%\\20% ratio of CP\\CI
#Save the simulations into one Excel file with tabs relating
#ENSURE that you change the file name specified in "EPSC_Simulation.py"
def create_all_simulations():
    glutamate_params = pd.DataFrame(
        [{"gl_mean": 3.5, "gl_sd": 1, "distribution_type": "lognormal", "fixed_value": None}])

    channel_params = pd.DataFrame(
        [{"distribution_type": "normal", "channel_sd": 560, "channel_mean": 1800, "fixed_value": None}])

    #Run this function four times, uncomment each one as we go and change filename in EPSC_Calc as we go
    # current_params = pd.DataFrame([{"iCP_Multiplier":3,"CP_Ratio":.1}])
    # current_params = pd.DataFrame([{"iCP_Multiplier":3,"CP_Ratio":.2}])
    # current_params = pd.DataFrame([{"iCP_Multiplier":6,"CP_Ratio":.1}])
    # current_params = pd.DataFrame([{"iCP_Multiplier":6,"CP_Ratio":.2}])
    ##Added in 0% on 4\\2\\2025
    # current_params = pd.DataFrame([{"iCP_Multiplier":6,"CP_Ratio":0}])
    current_params = pd.DataFrame([{"iCP_Multiplier":3,"CP_Ratio":0}])



    EPSCs_df, all_total_channel_nums, channel_data_df = EPSC_Calc(num_EPSCs=1000,channel_params=channel_params,glutamate_params=glutamate_params,current_params=current_params)

##STEP 2: Create a combined matrix and run NSFA
def NSFA_multiplier_CP_Ratios():
    ##Create a combined matrix
    output_file = 'multipliers_ratios.xlsx'
    folder_path = "C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\Scripts\\Experiments\\CP_CI_Ratio_Investigation\\EPSC_Variations"

    with pd.ExcelWriter(output_file) as writer:
            for file in os.listdir(folder_path):
                if file.endswith(".xlsx"):
                    file_path = os.path.join(folder_path,file)
                    df = pd.read_excel(file_path)
                    # Write the file to an Excel sheet
                    #Remove xlsx
                    df.to_excel(writer, sheet_name=str(file)[7:], index=False, header=False) #7: is to make the name <=31 chars


    params = {
        "alignment": ["peak"],
        "analysis_start_point": ["peak_start"],
        "scaling": ["minimize_error"],
        "output": ["linear", "parabolic"],
        "file_name": 'multipliers_ratios.xlsx',
        "folder_name": "C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\Scripts\\Experiments\\CP_CI_Ratio_Investigation\\NSFA_Results"

    }

    multiplier_ratio_matrix = matrix_generator(params,first_sheet=False)
    matrix_df = pd.DataFrame(multiplier_ratio_matrix)
    matrix_df.to_excel('multiplier_ratio_matrix.xlsx')


def create_fixed_simulation():
    glutamate_params = pd.DataFrame(
        [{"gl_mean": 3.5, "gl_sd": 1, "distribution_type": "lognormal", "fixed_value": None}])

    channel_params = pd.DataFrame(
        [{"distribution_type": "fixed_value", "channel_sd": None, "channel_mean": None, "fixed_value": 1000}])

    current_params_list = []
    current_params_list.append(pd.DataFrame([{"iCP_Multiplier":3,"CP_Ratio":.1}]))
    current_params_list.append(pd.DataFrame([{"iCP_Multiplier":3,"CP_Ratio":.2}]))
    current_params_list.append(pd.DataFrame([{"iCP_Multiplier":6,"CP_Ratio":.1}]))
    current_params_list.append(pd.DataFrame([{"iCP_Multiplier":6,"CP_Ratio":.2}]))
    current_params_list.append(pd.DataFrame([{"iCP_Multiplier":6,"CP_Ratio":0}]))
    current_params_list.append(pd.DataFrame([{"iCP_Multiplier":3,"CP_Ratio":0}]))

    for current_param in current_params_list:
        EPSCs_df, all_total_channel_nums, channel_data_df = EPSC_Calc(num_EPSCs=1000,channel_params=channel_params,
        glutamate_params=glutamate_params,current_params=current_param,  output_file_path = f'C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\Scripts\\Experiments\\CP_CI_Ratio_Investigation\\Fixed_CP_{current_param["CP_Ratio"][0]}-Mult-{current_param["iCP_Multiplier"][0]}.xlsx')


##INTERMEDIATE STEP: Creating CDFs of the EPSCs decay taus with the *median* rather than the mean so that Jim can fit the simulation parameters to that 
def create_median_tau_CDFs():
    file_name = 'C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\data-files\\Many_EPSCs_Juan.xlsx'
    folder_name = 'C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\Scripts\\Experiments\\CP_CI_Ratio_Investigation\\Median_CDF_Results\\'
    egg.generate_meanECDF_data(file_name=file_name,folder_name=folder_name,use_mean =True, use_median=True,plt_show=True,time=6)

def get_global_median_tau():
    file_name = 'C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\data-files\\Many_EPSCs_Juan.xlsx'
    folder_name = 'C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\Scripts\\Experiments\\CP_CI_Ratio_Investigation\\Median_CDF_Results\\'

    all_taus = []
    for name, sheet in pd.read_excel(file_name, sheet_name=None).items():
        print(sheet.shape,name)
        taus_array = egg.tau_graph_generator(sheet,folder_name,False,time=6)
        all_taus.append(taus_array)
    all_taus = pd.DataFrame(all_taus)
    all_taus.to_excel('C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\Scripts\\Experiments\\CP_CI_Ratio_Investigation\\Median_CDF_Results\\all_taus.xlsx')
    #Opened this and used Excel to get the median of the taus


#STEP 3: Run NSFA and plot the theoretical curves 


#STEP 4: Run the NSFA and plot the curves generated when the agonist is not a step function (stays open)
def EPSCs_steady_agonist(): #Essentially a duplicate of fixed EPSC function but with a steady agonist
    glutamate_params = pd.DataFrame(
        [{"gl_mean": 3.5, "gl_sd": 1, "distribution_type": "lognormal", "fixed_value": None}])

    channel_params = pd.DataFrame(
        [{"distribution_type": "fixed_value", "channel_sd": None, "channel_mean": None, "fixed_value": 1000}])

    current_params_list = []
    current_params_list.append(pd.DataFrame([{"iCP_Multiplier":3,"CP_Ratio":.1}]))
    current_params_list.append(pd.DataFrame([{"iCP_Multiplier":3,"CP_Ratio":.2}]))
    current_params_list.append(pd.DataFrame([{"iCP_Multiplier":6,"CP_Ratio":.1}]))
    current_params_list.append(pd.DataFrame([{"iCP_Multiplier":6,"CP_Ratio":.2}]))
    current_params_list.append(pd.DataFrame([{"iCP_Multiplier":6,"CP_Ratio":0}]))
    current_params_list.append(pd.DataFrame([{"iCP_Multiplier":3,"CP_Ratio":0}]))

    for current_param in current_params_list:
        EPSCs_df, all_total_channel_nums, channel_data_df = EPSC_Calc(num_EPSCs=1000,channel_params=channel_params,
        glutamate_params=glutamate_params,current_params=current_param,  output_file_path = f'C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\Scripts\\Experiments\\CP_CI_Ratio_Investigation\\EPSCS_Steady_Agonist_Fixed_n\\R-{current_param["CP_Ratio"][0]}-Mult-{current_param["iCP_Multiplier"][0]}.xlsx',
        folder_path='C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\Scripts\\Experiments\\CP_CI_Ratio_Investigation\\EPSCS_Steady_Agonist_Fixed_n',agonist_steady=True)




def EPSCs_Double_Agonist():
    glutamate_params = pd.DataFrame(
        [{"gl_mean": 3.5, "gl_sd": 1, "distribution_type": "lognormal", "fixed_value": None}])

    channel_params = pd.DataFrame(
        [{"distribution_type": "fixed_value", "channel_sd": None, "channel_mean": None, "fixed_value": 1000}])

def NSFA_steady_agonist():
        ##Create a combined matrix
        output_file = 'steady_EPSCs.xlsx'
        folder_path = 'EPSCS_Steady_Agonist_Fixed_n'
        with pd.ExcelWriter(output_file) as writer:
            for file in os.listdir(folder_path):
                if file.endswith(".xlsx"):
                    file_path = os.path.join(folder_path, file)
                    df = pd.read_excel(file_path)
                    # Write the file to an Excel sheet
                    # Remove xlsx
                    df.to_excel(writer, sheet_name=str(file), index=False,
                                header=False)  # 7: is to make the name <=31 chars

        params = {
            "alignment": ["peak"],
            "analysis_start_point": ["peak_start"],
            "scaling": ["minimize_error"],
            "output": ["linear", "parabolic"],
            "file_name": output_file,
            "folder_name": folder_path

        }

        matrix = matrix_generator(params, first_sheet=False)
        matrix_df = pd.DataFrame(matrix)
        matrix_df.to_excel('steady_agonists_matrix.xlsx')

        current_params_list = []
        current_params_list.append(pd.DataFrame([{"iCP_Multiplier":3,"CP_Ratio":.1}]))
        current_params_list.append(pd.DataFrame([{"iCP_Multiplier":3,"CP_Ratio":.2}]))
        current_params_list.append(pd.DataFrame([{"iCP_Multiplier":6,"CP_Ratio":.1}]))
        current_params_list.append(pd.DataFrame([{"iCP_Multiplier":6,"CP_Ratio":.2}]))
        current_params_list.append(pd.DataFrame([{"iCP_Multiplier":6,"CP_Ratio":0}]))
        current_params_list.append(pd.DataFrame([{"iCP_Multiplier":3,"CP_Ratio":0}]))

        for current_param in current_params_list:
            EPSCs_df, all_total_channel_nums, channel_data_df = EPSC_Calc(num_EPSCs=100,channel_params=channel_params,
            glutamate_params=glutamate_params,current_params=current_param,  output_file_path = f'C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\Scripts\\Experiments\\CP_CI_Ratio_Investigation\\EPSCS_Double_Agonist\\R-{current_param["CP_Ratio"][0]}-Mult-{current_param["iCP_Multiplier"][0]}.xlsx',
            folder_path='C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\Scripts\\Experiments\\CP_CI_Ratio_Investigation\\EPSC_Double_Agonist',agonist_steady=False,double_agonist=True)


def NSFA_double_agonist():
    ##Create a combined matrix
    output_file = 'double_agonist_epscs.xlsx'
    folder_path = 'EPSCS_Double_Agonist'
    with pd.ExcelWriter(output_file) as writer:
        for file in os.listdir(folder_path):
            if file.endswith(".xlsx"):
                file_path = os.path.join(folder_path, file)
                df = pd.read_excel(file_path)
                # Write the file to an Excel sheet
                # Remove xlsx
                df.to_excel(writer, sheet_name=str(file), index=False,
                            header=False)  # 7: is to make the name <=31 chars

    params = {
        "alignment": ["peak"],
        "analysis_start_point": ["peak_start"],
        "scaling": ["minimize_error"],
        "output": ["linear", "parabolic"],
        "file_name": output_file,
        "folder_name": folder_path

    }

    matrix = matrix_generator(params, first_sheet=False)
    matrix_df = pd.DataFrame(matrix)
    matrix_df.to_excel('double_agonist_matrix.xlsx')
def Test_Double_Agonist():
    ag_pulse = Agonist_Pulse(1,steady=False,second_pulse=True)[0:800]
    plt.figure()
    time = np.linspace(0,16,800)
    plt.xlabel("Time (ms)")
    plt.ylabel("Glutamate (mM)")

    plt.plot(time,ag_pulse/1000)
    plt.show()


'''The purpose of this function is to find if there is a specific number of EPSCs
required for the "i" estimate to be accurate with different CI/CP ratios'''
def EPSC_Size_Requirements():

    #Simulate 10000 EPSCs
    glutamate_params = pd.DataFrame(
        [{"gl_mean": 3.5, "gl_sd": 1, "distribution_type": "lognormal", "fixed_value": None}])

    channel_params = pd.DataFrame(
        [{"distribution_type": "fixed_value", "channel_sd": None, "channel_mean": None, "fixed_value": 1000}])

    current_params_list = []

    current_params_list.append(pd.DataFrame([{"iCP_Multiplier": 0, "CP_Ratio": 0}]))

    for current_param in current_params_list:

        EPSCs_df, all_total_channel_nums, channel_data_df = EPSC_Calc(num_EPSCs=5000, channel_params=channel_params,
                                                                      glutamate_params=glutamate_params,
                                                                      current_params=current_param,
                                                                      output_file_path=f'C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\Scripts\\Experiments\\CP_CI_Ratio_Investigation\\EPSC_Size_Investigation\\Fixed_CP_{current_param["CP_Ratio"][0]}-Mult-{current_param["iCP_Multiplier"][0]}.xlsx')


def NSFA_Size_Requirements(): #Running NSFA on 100, 250, and 500 samples of EPSCs from a 5000 EPSC file.
    sample_size_list = [25,50,100,250,500,1000]

    ##Create a combined matrix
    output_file = 'size_invest_EPSCs.xlsx'
    folder_path = 'EPSC_Size_Investigation'

    if os.path.isfile(output_file):
        print("Output file exists, reading current")
        df = pd.read_excel(output_file)
    else:
        with pd.ExcelWriter(output_file) as writer:
            for file in os.listdir(folder_path):
                if file.endswith(".xlsx"):
                    file_path = os.path.join(folder_path, file)
                    df = pd.read_excel(file_path)
                    # Write the file to an Excel sheet
                    # Remove xlsx
                    df.to_excel(writer, sheet_name=str(file), index=False,
                                header=False)  # 7: is to make the name <=31 chars

    sample_size_current_tracking = {}
    for sample_size in sample_size_list:
        sample_size_current_tracking[sample_size] = []
        for trial in range(0,20):
            #Select random columns
            sample = df.sample(sample_size, axis=1)
            print(sample.shape)
            params = {
                "direct_df_input":True,
                "sheet_names":[f"S{sample_size}T{trial}"],
                "EPSCs":sample,
                "alignment": ["peak"],
                "analysis_start_point": ["peak_start"],
                "scaling": ["minimize_error"],
                "output": ["linear", "parabolic"],
                "file_name": output_file,
                "folder_name": folder_path
            }
            matrix = matrix_generator(params, first_sheet=True,save_figs = False)
            i = matrix[0]["linear_i"]
            sample_size_current_tracking[sample_size].append(i)
    print(sample_size_current_tracking)
    sample_current_df  = pd.DataFrame(sample_size_current_tracking)
    sample_current_df.to_excel("Sample_Current_DF_first2.xlsx")

'''The purpose of this function is to find if there is a specific number of EPSCs
required for the "i" estimate to be accurate with different CI/CP ratios FOR CDF FIT simulation paramters'''
def EPSC_Size_Requirements_CDF_Fit():
    glutamate_params = pd.DataFrame(
        [{"gl_mean": 3.5, "gl_sd": 1, "distribution_type": "lognormal", "fixed_value": None}])

    channel_params = pd.DataFrame(
        [{"distribution_type": "normal", "channel_sd": 560, "channel_mean": 1800, "fixed_value": None}])

    current_params = pd.DataFrame([{"iCP_Multiplier": 0, "CP_Ratio": 0}])

    output_file_path = f'C:/Users/jawad/Downloads/Python-EPSC-NSFA-Pipeline/Scripts/Experiments/CP_CI_Ratio_Investigation/EPSCS_CDFfit_5000.xlsx'

    EPSCs_df, all_total_channel_nums, channel_data_df = EPSC_Calc(num_EPSCs=5000, channel_params=channel_params,
                                                                  glutamate_params=glutamate_params,
                                                                  current_params=current_params,
                                                                  output_file_path=output_file_path)


def NSFA_Size_Requirements_CDF_Fit(): #Running NSFA on 100, 250, and 500 samples of EPSCs from a 5000 EPSC file.
    sample_size_list = [25,50,100,250,500,1000]

    ##Create a combined matrix
    output_file = 'size_invest_EPSCs_cdf_fit.xlsx'
    folder_path = 'EPSC_Size_Invest_CDF_Fit'

    if os.path.isfile(output_file):
        print("Output file exists, reading current")
        df = pd.read_excel(output_file)
    else:
        with pd.ExcelWriter(output_file) as writer:
            for file in os.listdir(folder_path):
                if file.endswith(".xlsx"):
                    file_path = os.path.join(folder_path, file)
                    df = pd.read_excel(file_path)
                    # Write the file to an Excel sheet
                    # Remove xlsx
                    df.to_excel(writer, sheet_name=str(file), index=False,
                                header=False)  # 7: is to make the name <=31 chars

    sample_size_current_tracking = {}
    for sample_size in sample_size_list:
        sample_size_current_tracking[sample_size] = []
        for trial in range(0,20):
            #Select random columns
            sample = df.sample(sample_size, axis=1)
            print(sample.shape)
            params = {
                "direct_df_input":True,
                "sheet_names":[f"S{sample_size}T{trial}"],
                "EPSCs":sample,
                "alignment": ["peak"],
                "analysis_start_point": ["peak_start"],
                "scaling": ["minimize_error"],
                "output": ["linear", "parabolic"],
                "file_name": output_file,
                "folder_name": folder_path
            }
            matrix = matrix_generator(params, first_sheet=True,save_figs = False)
            i = matrix[0]["linear_i"]
            sample_size_current_tracking[sample_size].append(i)
    print(sample_size_current_tracking)
    sample_current_df  = pd.DataFrame(sample_size_current_tracking)
    sample_current_df.to_excel("Sample_Current_DF_CDF_fit.xlsx")
def median_cell_template_extraction():  #Running template extraction for all analysis types on Cell O23A
    templates = {}
    file_path = 'C:/Users/jawad/Downloads/Python-EPSC-NSFA-Pipeline/data-files/Many_EPSCs_Juan.xlsx'
    epscs = pd.read_excel(file_path,sheet_name='JG21O23A')
    time_duration = epscs.shape[0] * .02
    num_samples = epscs.shape[0]
    epscs = epscs.to_numpy()
    print(f"Shape of EPSCs: {epscs.shape}")
    print(epscs[:,1])
    timepoints = np.linspace(0, time_duration, num_samples)
    params = {
        "alignment":["peak","midpoint","max_dv_dt"]
    }
    plt.show()
    for alignment_type in params["alignment"]:
        if alignment_type == "peak":
            alignment_processed, peak_index = EPSC_preprocessing.align_peaks(epscs)
            start_point = peak_index
        elif alignment_type == "midpoint":
            alignment_processed = EPSC_preprocessing.align_midpoint(epscs)
            start_point = 15
        elif alignment_type == "max_dv_dt":
            alignment_processed = EPSC_preprocessing.align_dv_dt(epscs)
            start_point = 15

        timepoints, template = EPSC_App_Connection.create_template(alignment_processed,time_duration,num_samples)
        templates[alignment_type] = template
    template_df = pd.DataFrame(templates)
    template_df.to_excel("Median_Cell_Templates.xlsx")

def EPSC_Pools_New_Invest(): #The purpose of this function is to create 1000 EPSCs that fit the CDFs
    glutamate_params = pd.DataFrame(
        [{"gl_mean": 3.5, "gl_sd": 1, "distribution_type": "lognormal", "fixed_value": None}])

    channel_params = pd.DataFrame(
        [{"distribution_type": "normal", "channel_sd": 560, "channel_mean": 1800, "fixed_value": None}])

    current_params = pd.DataFrame([{"iCP_Multiplier":0,"CP_Ratio":0}])

    output_file_path = f'C:/Users/jawad/Downloads/Python-EPSC-NSFA-Pipeline/Scripts/Experiments/CP_CI_Ratio_Investigation/EPSCS_CDFfit_1000.xlsx'

    EPSCs_df, all_total_channel_nums, channel_data_df = EPSC_Calc(num_EPSCs=1001,channel_params=channel_params,glutamate_params=glutamate_params,current_params=current_params,output_file_path=output_file_path)

def EPSC_Size_Pools_Preprocessing():
    #End goal: have a DF with size pool ID, the amplitude range represented by that, the # of EPSCs in that pool,
    #the estimate of N for that size pool, and the true mean N of all of the EPSCs in that pool

    #Combine dataframes so that we have EPSCs + Open Channels
    EPSCs = pd.read_excel(f'C:/Users/jawad/Downloads/Python-EPSC-NSFA-Pipeline/Scripts/Experiments/CP_CI_Ratio_Investigation/EPSCS_CDFfit_1000.xlsx',sheet_name=0)
    channel_data = pd.read_excel(f'C:/Users/jawad/Downloads/Python-EPSC-NSFA-Pipeline/Scripts/Experiments/CP_CI_Ratio_Investigation/EPSCS_CDFfit_1000.xlsx',sheet_name=1)
    headers = {'EPSC': [], 'Open Channels': []}
    EPSC_df = pd.DataFrame(headers)

    for i in range(EPSCs.shape[1]):
        epsc = np.array(EPSCs.iloc[:, i])  # Assuming EPSCs has multiple columns of data
        open_channels = channel_data['Open Channels'][i]
        new_row = pd.DataFrame({'EPSC': [epsc], 'Open Channels': [open_channels]})
        EPSC_df = pd.concat([EPSC_df, new_row], ignore_index=True)

    #Sort the dataframe by the size of the EPSC maximum point. Create a column for "EPSC max". Values are descending
    EPSC_df['EPSC max'] = EPSC_df['EPSC'].apply(np.max)
    EPSC_df_sorted = EPSC_df.sort_values(by='EPSC max', ascending=False).reset_index(drop=True)


    #Define the bins as max-min EPSC/10
    num_bins = 10
    bins = pd.cut(EPSC_df_sorted['EPSC max'], bins=10, retbins=True)[1]

    # EPSC amplitude range (includes the interval labels)
    EPSC_df_sorted['EPSC amplitude range'] = pd.cut(EPSC_df_sorted['EPSC max'], bins=bins)

    # Bin numbers (0 to 9)
    EPSC_df_sorted['EPSC bin number'] = pd.cut(EPSC_df_sorted['EPSC max'], bins=bins, labels=False)

    #Columns for EPSC_df_sorted: Index(['EPSC', 'Open Channels', 'EPSC max', 'EPSC bin label'], dtype='object','EPSC bin number')

    ##########Begin NSFA ############

    ##Create a combined matrix
    output_file = 'size_pool_invest.xlsx'
    folder_path = 'EPSC_Pooling_Invest'

    if os.path.isfile(output_file):
        print("Output file exists, reading current")
        df = pd.read_excel(output_file)
    else:
        with pd.ExcelWriter(output_file) as writer:
            for file in os.listdir(folder_path):
                if file.endswith(".xlsx"):
                    file_path = os.path.join(folder_path, file)
                    df = pd.read_excel(file_path)
                    # Write the file to an Excel sheet
                    # Remove xlsx
                    df.to_excel(writer, sheet_name=str(file), index=False,
                                header=False)

    matrices = []
    for bin_idx in range(0,num_bins):
        # Get the list of EPSC arrays for a specific bin
        epsc_arrays = EPSC_df_sorted.loc[EPSC_df_sorted['EPSC bin number'] == bin_idx, 'EPSC'].tolist()
        # Convert to 2D NumPy array and then to DataFrame
        epsc_matrix = np.column_stack(epsc_arrays)  # shape: (time_points, num_traces)
        epsc_bin_df = pd.DataFrame(epsc_matrix)
        print(epsc_bin_df.shape)
        params = {
                "direct_df_input":True,
                "sheet_names":[f"Pool {bin_idx}"],
                "EPSCs":epsc_bin_df,
                "alignment": ["peak"],
                "analysis_start_point": ["peak_start"],
                "scaling": ["minimize_error"],
                "output": ["linear", "parabolic"],
                "file_name": output_file,
                "folder_name": folder_path
            }
        matrix = matrix_generator(params, first_sheet=True,save_figs = False)
        matrices.append(matrix[0])

    summary_data = []
    print(matrices)
    # Normalize and combine
    flat_matrices = [json_normalize(m) for m in matrices]
    matrices_df = pd.concat(flat_matrices, ignore_index=True)

    # Save to Excel
    matrices_df.to_excel("NSFA_5_pools.xlsx", index=False)

    for bin_idx in range(0,num_bins):
        # EPSCs in the bin
        bin_df = EPSC_df_sorted[EPSC_df_sorted['EPSC bin number'] == bin_idx]

        # EPSC count
        epsc_count = len(bin_df)

        # Mean Open Channels for this bin
        mean_open_channels = bin_df["Open Channels"].mean() if epsc_count > 0 else np.nan
        amplitude_range = bin_df['EPSC amplitude range'].unique()[0]  # Assuming only one unique range per bin
        # Estimated N from matrix result
        estimated_n = matrices[bin_idx]["num_channels"]

        # Append data row
        summary_data.append({
            "EPSC bin number": bin_idx,
            "Amplitude Range": amplitude_range,
            "Estimated N": estimated_n,
            "Mean Open Channels": mean_open_channels
        })

    # Create DataFrame
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel("EPSC_bin_summary.xlsx", index=False)
    # Convert the Amplitude Range intervals to strings
    # Set "Estimated N" to 0 if it's greater than 1000
    summary_df.loc[summary_df["Estimated N"] > 1000, "Estimated N"] = 0

    # Convert the Amplitude Range intervals to strings
    summary_df['Amplitude Range'] = summary_df['Amplitude Range'].astype(str)

    # Calculate the number of EPSCs per bin (count how many EPSCs are in each bin)
    # We can use groupby and count
    epsc_count = EPSC_df_sorted.groupby('EPSC bin number')['EPSC'].count().values
    print(epsc_count)
    # Add this count to the summary dataframe to use as the color
    summary_df['EPSC Count'] = epsc_count

    # Now, we plot the data, with colors based on the number of EPSCs
    plt.close('all')

    # Scatter plot where the color of the points depends on the 'EPSC Count' column
    plt.scatter(summary_df["Amplitude Range"], summary_df["Estimated N"],
                c=summary_df['EPSC Count'], cmap='viridis', marker='o', label='Estimated N')

    # Optional: You can also color the 'Mean Open Channels' data similarly
    plt.scatter(summary_df["Amplitude Range"], summary_df["Mean Open Channels"],
                c=summary_df['EPSC Count'], cmap='viridis', marker='x', label='Mean Open Channels')

    plt.xlabel("EPSC Bin Number")
    plt.ylabel("Channel Estimate")
    plt.title("Estimated vs. Mean Open Channels by Bin")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)  # Optional, to make interval labels more readable

    # Ensure the plot fits without cutting off any elements
    plt.tight_layout()

    # Show the plot
    plt.show()
    #
    #

    #For every bin, extract the EPSCs
        #DF: Size Bin, n-mean, # of EPSCs
        #Create a column for "n estimated" and populate it for every bin number
        #Add it to a subplot


    #Save the DF as an Excel File

    #Show plot






    return

def NSFA_Size_Requirements_Physiological(): #Running NSFA on 100, 250, and 500 samples of EPSCs from the physiological EPSC file
    sample_size_list = [25,50,100,250,500,1000]

    ##Create a combined matrix
    output_file = 'size_invest_EPSCs_physio.xlsx'
    folder_path = 'Physio_Invest'
    df = pd.read_excel("C:/Users/jawad/Downloads/Python-EPSC-NSFA-Pipeline/data-files/Many_EPSCs_Juan.xlsx",sheet_name='JG21O23A')

    sample_size_current_tracking = {}
    for sample_size in sample_size_list:
        sample_size_current_tracking[sample_size] = []
        for trial in range(0,20):
            #Select random columns
            sample = df.sample(sample_size, axis=1,replace=True)
            print(sample.shape)
            params = {
                "direct_df_input":True,
                "sheet_names":[f"S{sample_size}T{trial}"],
                "EPSCs":sample,
                "alignment": ["peak"],
                "analysis_start_point": ["peak_start"],
                "scaling": ["minimize_error"],
                "output": ["linear", "parabolic"],
                "file_name": output_file,
                "folder_name": folder_path
            }
            matrix = matrix_generator(params, first_sheet=True,save_figs = False)
            i = matrix[0]["linear_i"]
            sample_size_current_tracking[sample_size].append(i)
    print(sample_size_current_tracking)
    sample_current_df  = pd.DataFrame(sample_size_current_tracking)
    sample_current_df.to_excel("Sample_Current_DF_physio.xlsx")


def create_ratio_gradient():

    ci_ratio_list = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]
    glutamate_params = pd.DataFrame(
        [{"gl_mean": 2.5, "gl_sd": 1, "distribution_type": "normal", "fixed_value": None}])

    channel_params = pd.DataFrame(
        [{"distribution_type": "normal", "channel_sd": 700, "channel_mean": 2200, "fixed_value": None}])

    for ratio in ci_ratio_list:
        ci_ratio = ratio
        cp_ratio = 1 - ci_ratio
        current_params = pd.DataFrame([{"iCP_Multiplier":3,"CP_Ratio":cp_ratio}])
        print(f"Creating EPSCs for CI: {ci_ratio} CP: {cp_ratio}")
        output_file_path = f'C:/Users/jawad/Downloads/Python-EPSC-NSFA-Pipeline/Scripts/Experiments/CP_CI_Ratio_Investigation/EPSC_Ratio_Gradient/epsc_grad_ci{int(ci_ratio*10)}_cp{int(cp_ratio*10)}.xlsx'
        EPSCs_df, all_total_channel_nums, channel_data_df = EPSC_Calc(num_EPSCs=1000,channel_params=channel_params,glutamate_params=glutamate_params,current_params=current_params,output_file_path=output_file_path)

def NSFA_ratio_gradient():
        ##Create a combined matrix
        output_file = 'epsc_ratio_gradient_comb_v2.xlsx'
        folder_path = "C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\Scripts\\Experiments\\CP_CI_Ratio_Investigation\\EPSC_Ratio_Gradient"

        with pd.ExcelWriter(output_file) as writer:
            for file in os.listdir(folder_path):
                if file.endswith(".xlsx"):
                    file_path = os.path.join(folder_path, file)
                    filename = file.split('.')[0]
                    print("Appending:", filename)
                    df = pd.read_excel(file_path)
                    # Write the file to an Excel sheet
                    # Remove xlsx
                    df.to_excel(writer, sheet_name=filename, index=False,
                                header=False)  # 30: is to make the name <=31 chars

        params = {
            "alignment":["peak","midpoint","max_dv_dt"],
            "direct_df_input": False,
            "analysis_start_point":["peak_start",],
            "scaling":["minimize_error","peak_scaling_at_peak_time","peak_to_peak_scaling"],
            "output": ["linear","parabolic"],
            "file_name": output_file,
            "folder_name": "C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\Scripts\\Experiments\\CP_CI_Ratio_Investigation\\NSFA_Ratio_Gradient",
            "recording_duration":16
        }

        multiplier_ratio_matrix = matrix_generator(params, first_sheet=False)
        matrix_df = pd.DataFrame(multiplier_ratio_matrix)
        matrix_df.to_excel('nsfa_ratio_gradient_v2.xlsx')

def visualize_nsfa_gradient():
    # Load data
    df = pd.read_excel("nsfa_ratio_gradient_v2.xlsx")

    # Add CP and CI as integers
    df['CP'] = df['ratio condition'].str[-1].astype(int)


    df['CI'] = df['ratio condition'].str.extract(r'ci(\d+)', expand=False).astype(int)


    # Compute numeric CP:CI ratio
    df['CP_CI_ratio_num'] = df['CP'] / df['CI']
    # Also create a string label for categorical display
    df['CP_CI_ratio_str'] = df['CP'].astype(str) + ":" + df['CI'].astype(str)

    scaling_types = ["minimize_error", "peak_scaling_at_peak_time"]
    alignment_types = df['alignment'].unique()

    sns.set(style="white")  # No grid background

    for scaling in scaling_types:
        for alignment in alignment_types:
            sub_df = df[(df['scaling'] == scaling) & (df['alignment'] == alignment)].copy()

            # Filter out invalid or infinite ratios (CI == 0 leads to inf)
            sub_df = sub_df[
                sub_df['CP_CI_ratio_num'].notna() &
                np.isfinite(sub_df['CP_CI_ratio_num']) &
                sub_df['linear_i'].notna() &
                np.isfinite(sub_df['linear_i'])
                ]

            plt.figure(figsize=(10, 5))

            # # Scatter (strip plot style, but numeric axis)
            sns.stripplot(
                data=sub_df,
                x='CP_CI_ratio_str',
                y='linear_i',
                jitter=False,
                color='blue',
            )

            # sns.regplot(
            #     data=sub_df,
            #     x='CP',
            #     y='linear_i',
            #     ci=False
            # )


            plt.title(f"linear_i by CP/CI Ratio\nScaling: {scaling}, Alignment: {alignment}")
            plt.xlabel("CP / CI Ratio")
            plt.ylabel("linear_i (pA)")
            plt.grid(False)
            plt.tight_layout()
            ax = plt.gca()
            ticks = ax.get_xticks()
            tick_labels = sub_df['CP_CI_ratio_str']
            ax.set_xticklabels(tick_labels, rotation=45)
            plt.savefig(f'Ratio_Gradient_Plots/ratio_scatter_v2/{scaling}-{alignment}-ratio-scatter.png')

            ici = sub_df['linear_i'].iloc[-1]
            ict_array = sub_df['linear_i'].iloc[0:-1]
            ict_array = ict_array.reset_index()['linear_i']
            B = ict_array - (ici/ict_array)# fraction "blocked"


            icp = ici + ((ict_array - ici)/(B))

            fCP = ici/((icp/B)-(icp - ici))

            # Ensure matching and clean indexes
            # Step 1: Slice and reset both Series cleanly
            true_cp = sub_df['CP'].iloc[0:-1].reset_index(drop=True)
            pred_cp = fCP.iloc[0:-1].reset_index(drop=True)

            # Step 2: Build a DataFrame to align everything
            cp_df = pd.DataFrame({
                'true_cp': true_cp,
                'pred_cp': pred_cp
            })


            # Step 3: Masks based on pred_cp values
            valid_mask = (cp_df['pred_cp'] <= 80) | (cp_df['pred_cp'] >= -10)
            invalid_mask = (cp_df['pred_cp'] > 80) | (cp_df['pred_cp'] < -10)

            # Step 4: Plot
            plt.figure(figsize=(6, 6))

            # Blue: valid predictions
            plt.scatter(cp_df.loc[valid_mask, 'true_cp']/10, cp_df.loc[valid_mask, 'pred_cp']/10,
                        c='blue')
            xmin, xmax = plt.xlim()  # get current limits
            plt.xlim(xmax, xmin)  # reverse them

            # Red X: invalid predictions, show at (true_cp, true_cp)
            if cp_df.loc[invalid_mask, 'true_cp'].any():
                x_vals = cp_df.loc[invalid_mask, 'true_cp']
                y_vals = cp_df.loc[invalid_mask, 'true_cp']  # plotted on diagonal
                outlier_vals = cp_df.loc[invalid_mask, 'pred_cp']

                # Scatter red Xs
                plt.scatter(x_vals/10, y_vals/10, c='red', marker='x', label='Outlier Point Removed')

                # Annotate each red X with the predicted value
                for x, y, val in zip(x_vals/10, y_vals/10, outlier_vals):
                    plt.annotate(f"{val:.1f}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9,
                                 color='red')
                plt.legend()

            # Labels and formatting
            plt.xlabel('True fCP')
            plt.ylabel('Predicted fCP')
            plt.title(f'Predicted vs True CP\nScaling: {scaling}, Alignment: {alignment}')
            plt.xlim([0,1])
            lin_x = np.linspace(0,1,10)
            lin_y = np.linspace(0,1,10)
            plt.plot(lin_x,lin_y)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'Ratio_Gradient_Plots/true_pred_cp_v2/{scaling}-{alignment}-cp-comp.png')


def calculate_fcp(ici,ict): #sub_df should be a dataframe/series with the column 'linear_i'

    B = ict - (ici / ict)  # fraction "blocked"

    icp = ici + ((ict - ici) / (B))

    fCP = ici / ((icp / B) - (icp - ici))

    return B,icp, fCP
def IEM_vs_Control_NSFA():

    iem_data_path = 'C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\data-files\\IEM_Data\\IEM_EPSCs.xlsx'
    control_data_path = 'C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\data-files\\Many_EPSCs_Juan.xlsx'

    print("Reading in Control Data...")
    control_data = pd.read_excel(control_data_path,sheet_name=None)
    print("Reading in IEM Data...")
    iem_data = pd.read_excel(iem_data_path,sheet_name=None)

    matrices = []

    for cell in control_data:
        for key in iem_data:
            if cell in key: #matching cells
                iem_epscs = iem_data[key]
                control_epscs = control_data[cell]

                #Run NSFA on both to extract iCP/iCT
                params = {
                    "EPSCs":iem_epscs,
                    "alignment": ["peak", "midpoint", "max_dv_dt"],
                    "direct_df_input": True,
                    "analysis_start_point": ["peak_start", ],
                    "scaling": ["minimize_error","peak_scaling_at_peak_time"],
                    "output": ["linear", "parabolic"],
                    "sheet_names": [key],
                    "folder_name": "C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\Scripts\\Experiments\\CP_CI_Ratio_Investigation\\NSFA_IEM_Control",
                    "recording_duration":6 #ms

                }

                multiplier_ratio_matrix = matrix_generator(params, first_sheet=False,debug=True)
                matrix_df = pd.DataFrame(multiplier_ratio_matrix)
                matrices.append(matrix_df)
                matrix_df.to_excel(f'nsfa_phys_{key}.xlsx')

                # Run NSFA on both to extract iCP/iCT
                params = {
                    "EPSCs": control_epscs,
                    "alignment": ["peak", "midpoint", "max_dv_dt"],
                    "direct_df_input": True,
                    "analysis_start_point": ["peak_start", ],
                    "scaling": ["minimize_error","peak_scaling_at_peak_time"],
                    "output": ["linear", "parabolic"],
                    "sheet_names": [cell],
                    "folder_name": "C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\Scripts\\Experiments\\CP_CI_Ratio_Investigation\\NSFA_IEM_Control",
                    "recording_duration":6
                }

                multiplier_ratio_matrix = matrix_generator(params, first_sheet=False)
                matrix_df = pd.DataFrame(multiplier_ratio_matrix)
                matrices.append(matrix_df)
                # matrix_df.to_excel(f'nsfa_phys_{cell}.xlsx')
    matrices_df = pd.concat(matrices, ignore_index=True)
    matrices_df.to_excel("nsfa_phys_full_v2.xlsx", index=False)


def fCP_Extract_Phys():
    matrix = pd.read_excel('nsfa_phys_full_v2.xlsx')
    results = []

    alignment_types = ["peak", "midpoint", "max_dv_dt"]
    scaling_types =  ["minimize_error","peak_scaling_at_peak_time"]

    for cell in matrix['cell_name'].unique():
        for alignment in alignment_types:
            for scaling in scaling_types:
                if 'IEM' in cell:
                    iem_row = matrix[(matrix['cell_name'] == cell) & (matrix['alignment'] == alignment) & (matrix['scaling'] == scaling)]
                    ici = iem_row['linear_i'].values[0]  # Extract scalar value
                    control_name = cell.split('_')[0]

                    control_row = matrix[(matrix['cell_name'] == control_name) & (matrix['alignment'] == alignment)& (matrix['scaling'] == scaling)]


                    if not control_row.empty:
                        ict = control_row['linear_i'].values[0]  # Extract scalar value

                        B, iCP, fCP = calculate_fcp(ici, ict)

                        results.append({
                            'IEM_cell': cell,
                            'Control_cell': control_name,
                            'Scaling':scaling,
                            'Alignment':alignment,
                            'B': B,
                            'iCP': iCP,
                            'fCP': fCP,
                            'ict':ict,
                            'ici':ici
                        })

    final_df = pd.DataFrame(results)
    final_df.to_excel("fcp_results_v3.xlsx", index=False)
    return final_df

def create_combined_figure():
    # Path to your folder
    folder = "NSFA_IEM_Control"

    pattern = re.compile(
        r"(?P<alignment>max_dv_dt|midpoint|peak)_"
        r"peak_start_"
        r"(?P<scaling>minimize_error|peak_scaling_at_peak_time)_"
        r"(?P<cell>JG[0-9A-Z]+)"
        r"(?:_IEM(?P<drug>\d+))?"
        r"\.png$"
    )

    cell_images = defaultdict(lambda: defaultdict(dict))

    for fname in os.listdir(folder):
        if not fname.endswith(".png"):
            continue
        m = pattern.match(fname)
        if not m:
            continue

        alignment = m.group("alignment")
        scaling = m.group("scaling")
        cell = m.group("cell")
        drug = m.group("drug")
        path = os.path.join(folder, fname)

        key = (alignment, scaling)
        if drug is None:
            cell_images[cell][key]["control"] = path
        else:
            cell_images[cell][key]["drug"] = (path, drug)  # store tuple (path, concentration)

    for cell, combos in cell_images.items():
        keys = sorted(combos.keys(), key=lambda x: (x[0], x[1]))
        ncols = len(keys)

        fig, axes = plt.subplots(2, ncols, figsize=(3 * ncols, 6), squeeze=False)
        fig.suptitle(f"Cell {cell}", fontsize=14)

        for col, (alignment, scaling) in enumerate(keys):
            # Control (top)
            if "control" in combos[(alignment, scaling)]:
                img = mpimg.imread(combos[(alignment, scaling)]["control"])
                axes[0, col].imshow(img)
                axes[0, col].set_title(f"{alignment}\n{scaling}", fontsize=8)
            axes[0, col].axis("off")

            # Drug (bottom)
            if "drug" in combos[(alignment, scaling)]:
                path, conc = combos[(alignment, scaling)]["drug"]
                img = mpimg.imread(path)
                axes[1, col].imshow(img)
            axes[1, col].axis("off")

        # Label the rows

        fig.text(0.005, 0.72, "Control", va="center", ha="left", fontsize=12, rotation=90)
        fig.text(0.005, 0.28, f"Drug ({conc}uM)", va="center", ha="left", fontsize=12, rotation=90)

        plt.tight_layout()
        plt.savefig(f'NSFA_IEM_Control/combined_figs/{cell}.png')
        # plt.show()

def create_ratio_gradient_variability():

    ci_ratio_list = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]
    glutamate_params = pd.DataFrame(
        [{"gl_mean": 2.5, "gl_sd": 1, "distribution_type": "normal", "fixed_value": None,"continuous":True}])

    channel_params = pd.DataFrame(
        [{"distribution_type": "normal", "channel_sd": 700, "channel_mean": 2200, "fixed_value": None}])

    for ratio in ci_ratio_list:
            ci_ratio = ratio
            cp_ratio = 1 - ci_ratio
            current_params = pd.DataFrame([{"iCP_Multiplier":3,"CP_Ratio":cp_ratio}])
            print(f"Creating EPSCs for CI: {ci_ratio} CP: {cp_ratio}")
            output_file_path = f'C:/Users/jawad/Downloads/Python-EPSC-NSFA-Pipeline/Scripts/Experiments/CP_CI_Ratio_Investigation/EPSC_Variability_Ratio_Gradients/epsc5000_grad_ci{int(ci_ratio*10)}_cp{int(cp_ratio*10)}.xlsx'
            EPSCs_df, all_total_channel_nums, channel_data_df = EPSC_Calc(num_EPSCs=5000,channel_params=channel_params,glutamate_params=glutamate_params,current_params=current_params,output_file_path=output_file_path)


def NSFA_ratio_gradient_variability():
    ##Create a combined matrix
    output_file = 'epsc_ratio_gradient_var.xlsx'
    folder_path = "C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\Scripts\\Experiments\\CP_CI_Ratio_Investigation\\EPSC_Variability_Ratio_Gradients"

    # with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
    #     for file in os.listdir(folder_path):
    #         if file.endswith(".xlsx"):
    #             file_path = os.path.join(folder_path, file)
    #             filename = os.path.splitext(file)[0]
    #             print(f"Processing: {filename}")
    #             # Open source file in streaming mode
    #             wb = load_workbook(file_path, read_only=True)
    #             ws = wb.active  # assuming first sheet
    #
    #             total_cols = ws.max_column
    #             total_rows = ws.max_row
    #
    #             for i in range(0, total_cols, 1000):
    #                 print(f"Iteration: {i//1000}")
    #                 # Collect one 1000-column chunk
    #                 data = []
    #                 for row in ws.iter_rows(min_row=1, max_row=total_rows,
    #                                         min_col=i + 1, max_col=min(i + 1000, total_cols),
    #                                         values_only=True):
    #                     data.append(row)
    #
    #                 df = pd.DataFrame(data)
    #
    #                 # Create sheet name
    #                 parts = filename.split("_")
    #                 ci_cp = "_".join(parts[-2:])  # e.g., "ci6_cp4"
    #                 sheet_name = f"{ci_cp}_{i // 1000}"[:31]  # Excel sheet names <=31 chars
    #
    #                 df.to_excel(writer, sheet_name=sheet_name,
    #                             index=False, header=False)
    #
    #             wb.close()


    params = {
        "alignment": ["peak", "midpoint", "max_dv_dt"],
        "direct_df_input": False,
        "analysis_start_point": ["peak_start", ],
        "scaling": ["minimize_error", "peak_scaling_at_peak_time", "peak_to_peak_scaling"],
        "output": ["linear", "parabolic"],
        "file_name": output_file,
        "folder_name": "C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\Scripts\\Experiments\\CP_CI_Ratio_Investigation\\NSFA_Variability_Ratio_Gradients",
        "recording_duration": 16
    }

    multiplier_ratio_matrix = matrix_generator(params, first_sheet=False,save_figs=False)
    matrix_df = pd.DataFrame(multiplier_ratio_matrix)
    matrix_df.to_excel('nsfa_var_ratio_gradient.xlsx')

def visualize_nsfa_gradient_var():
    # Load data
    df = pd.read_excel("nsfa_var_ratio_gradient.xlsx")

    # Split out condition vs run
    df['run'] = df['ratio condition'].str.extract(r'_(\d+)$').astype(int)
    df['condition'] = df['ratio condition'].str.replace(r'_\d+$', '', regex=True)

    # Add CP and CI as integers
    df['CP'] = df['condition'].str.extract(r'cp(\d+)', expand=False).astype(int)
    df['CI'] = df['condition'].str.extract(r'ci(\d+)', expand=False).astype(int)

    # Compute numeric CP:CI ratio
    df['CP_CI_ratio_num'] = df['CP'] / df['CI'].replace(0, np.inf)
    df['CP_CI_ratio_str'] = df['CP'].astype(str) + ":" + df['CI'].astype(str)

    scaling_types = ["minimize_error", "peak_scaling_at_peak_time"]
    alignment_types = df['alignment'].unique()

    sns.set(style="white")  # No grid background
    tables = []
    for scaling in scaling_types:
        for alignment in alignment_types:
            sub_df = df[(df['scaling'] == scaling) & (df['alignment'] == alignment)].copy()

            # Filter valid rows
            sub_df = sub_df[
                sub_df['CP_CI_ratio_num'].notna() &
                sub_df['linear_i'].notna() &
                np.isfinite(sub_df['linear_i'])
            ]



            # --- collapse runs: take mean and std per condition ---
            grouped = sub_df.groupby(
                ['condition', 'CP', 'CI', 'CP_CI_ratio_str'], as_index=False
            ).agg(
                mean_linear_i=('linear_i', 'mean'),
                min_linear_i=('linear_i', 'min'),
                max_linear_i=('linear_i', 'max')
            )

            # Compute range as asymmetric error bars
            grouped['err_lower'] = grouped['mean_linear_i'] - grouped['min_linear_i']
            grouped['err_upper'] = grouped['max_linear_i'] - grouped['mean_linear_i']

            # Sort numerically
            grouped = grouped.sort_values(['CI', 'CP']).reset_index(drop=True)

            # --- plot scatter with error bars ---
            plt.figure(figsize=(10, 5))
            plt.errorbar(
                x=grouped['CP_CI_ratio_str'],
                y=grouped['mean_linear_i'],
                yerr=[grouped['err_lower'], grouped['err_upper']],  # asymmetric error bars
                fmt='o', color='blue', ecolor='black', capsize=4
            )
            for i, row in grouped.iterrows():
                plt.text(
                    i + 0.1,  # horizontal shift: +0.1 to the right
                    row['mean_linear_i'] + 0.02,  # vertical shift: +0.02 upward
                    f"{row['mean_linear_i']:.2f}",
                    ha='left', va='bottom', fontsize=9, color='black'
                )
            plt.title(f"linear_i by CP/CI Ratio\nScaling: {scaling}, Alignment: {alignment}")
            plt.xlabel("CP / CI Ratio")
            plt.ylabel("linear_i (pA)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'Ratio_Gradient_Plots/ratio_scatter_var/{scaling}-{alignment}-scatter-var.png')


            #######Plotting based on cell fraction equation
            ici = grouped['mean_linear_i'].iloc[-1]   #Mean of the CI10_CP0 state, should be an integer
            ict_array = sub_df['linear_i'] #Series
            ict_array = ict_array.reset_index()['linear_i']
            B = (ict_array - ici)/ict_array# fraction of current "blocked"




            icp = ici + ((ict_array - ici)/(B))
            #FCP is not as long as pred_Cp
            fCP = ici/((icp/B)-(icp - ici))






            true_cp = sub_df['CP']
            pred_cp = fCP
            true_cp.reset_index(drop=True, inplace=True)
            pred_cp.reset_index(drop=True, inplace=True)
            ###Debugging
            table =pd.DataFrame({
                "scaling": scaling,
                "alignment": alignment,
                "ici": ici,  # scalar -> repeats for every row
                "ict_array": ict_array,
                "B": B,
                "icp": icp,
                "pred_cp": fCP,
                "true_cp": true_cp/10,
                "error":((true_cp)/10)- fCP
            })
            tables.append(table)

            # Optional: round values for readability
            # table = table.round(4)

            # Print as a clean table
            # print(table.to_string(index=False))

            # Step 4: Plot
            plt.close('all')

            plt.figure(figsize=(6, 6))

            error = ((true_cp)/10) - fCP

            plt.scatter(true_cp/10,error)

            xmin, xmax = plt.xlim()  # get current limits
            plt.xlim(xmax, xmin)  # reverse them


            plt.xlabel('True fCP')
            plt.ylabel('Error (True-Pred CP)')
            plt.title(f'Predicted vs True CP\nScaling: {scaling}, Alignment: {alignment}')
            plt.xlim([-.01,1.01])
            lin_x = np.linspace(0,1,10)
            lin_y = np.linspace(0,1,10)
            plt.plot(lin_x,lin_y)
            plt.grid(True)
            # plt.xlabel("True fCP")
            plt.tight_layout()
            # plt.show()
            plt.savefig(f'Ratio_Gradient_Plots/true_pred_cp_var/{scaling}-{alignment}-cp-comp.png')

        tables_df = pd.concat(tables)
        tables_df.to_excel("cp_invest.xlsx")
        # load the sample data
        df = pd.DataFrame({'MutProb': [0.1,
                                       0.05, 0.01, 0.005, 0.001, 0.1, 0.05, 0.01, 0.005, 0.001, 0.1, 0.05, 0.01, 0.005,
                                       0.001, 0.1, 0.05, 0.01, 0.005, 0.001, 0.1, 0.05, 0.01, 0.005, 0.001],
                           'SymmetricDivision': [1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.6, 0.6, 0.6, 0.6,
                                                 0.6, 0.4, 0.4, 0.4, 0.4, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2],
                           'test': ['sackin_yule', 'sackin_yule', 'sackin_yule', 'sackin_yule', 'sackin_yule',
                                    'sackin_yule', 'sackin_yule', 'sackin_yule', 'sackin_yule', 'sackin_yule',
                                    'sackin_yule', 'sackin_yule', 'sackin_yule', 'sackin_yule', 'sackin_yule',
                                    'sackin_yule', 'sackin_yule', 'sackin_yule', 'sackin_yule', 'sackin_yule',
                                    'sackin_yule', 'sackin_yule', 'sackin_yule', 'sackin_yule', 'sackin_yule'],
                           'value': [-4.1808639999999997, -9.1753490000000006, -11.408113999999999, -10.50245,
                                     -8.0274750000000008, -0.72260200000000008, -6.9963940000000004,
                                     -10.536339999999999, -9.5440649999999998, -7.1964070000000007,
                                     -0.39225599999999999, -6.6216390000000001, -9.5518009999999993,
                                     -9.2924690000000005, -6.7605589999999998, -0.65214700000000003,
                                     -6.8852289999999989, -9.4557760000000002, -8.9364629999999998, -6.4736289999999999,
                                     -0.96481800000000006, -6.051482, -9.7846860000000007, -8.5710630000000005,
                                     -6.1461209999999999]})

        # pivot the dataframe from long to wide form
        pivot = tables_df.pivot(index='alignment', columns='scaling', values='error')

        sns.heatmap(pivot, annot=True, fmt="g", cmap='viridis')
        plt.show()




if __name__ == "__main__":
    # print(f"Directory {sys.path}")
    # create_all_simulations()
    # create_fixed_simulation()
    # NSFA_multiplier_CP_Ratios()
    # create_median_tau_CDFs()
    # get_global_median_tau()
    # EPSCs_steady_agonist() #Had to rename xlsx files because .1 was being split during the matrix generator
    # NSFA_steady_agonist()
    # Test_Double_Agonist()
    # EPSCs_Double_Agonist()
    # NSFA_double_agonist()
    # EPSC_Size_Requirements()
    # NSFA_Size_Requirements()
    # median_cell_template_extraction()
    #Changed linear slope to fit to first two points in EPSC_App_Connection parabolic
    #Changed file name output of NSFA_Size_Requirements to reflect "first2"
    # NSFA_Size_Requirements()

    ###Investigating how NSFA estimates of N change with size pools
    #EPSC_Pools_New_Invest()
    # EPSC_Size_Pools_Preprocessing()

    ###Investigating if the size pool accuracy stays the same for the CDF fit simulation parameters
    # EPSC_Size_Requirements_CDF_Fit()
    # NSFA_Size_Requirements_CDF_Fit()

    #And for physiological JG21O23A
    # NSFA_Size_Requirements_Physiological()


    #Now we are looking to create EPSC datasets for increments of 10 for CP/CI
    # create_ratio_gradient()


    #And to look at NSFA matrices for each one of those new files.
    # NSFA_ratio_gradient()

    #then to visualize them and the predicted ratios. To make it fit, I changed the first column name to "ratio gradient"
    #I also had to re-order the cell ratios to put CP10 at the end manually.
    # visualize_nsfa_gradient()

    #Now, it's time to do the same for the physiological data.
    # We need to extract iCI by running NSFA on the IEM conditions
    #and iCT by running NSFA on the control conditions
    # IEM_vs_Control_NSFA()
    # fCP_Extract_Phys()

    #helper function for visualizing by cell
    # create_combined_figure()

    #We now want to see how much variation there is in the simulations for the predicted current and predicted fCP
    create_ratio_gradient_variability()
    # NSFA_ratio_gradient_variability()
    # visualize_nsfa_gradient_var()

    #New question: for our CDF fit data, is there an increase in 1.) variability and 2.)