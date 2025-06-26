import os
import sys
# current_dir = os.path.dirname(os.path.abspath(__file__))
# scripts_dir = os.path.abspath(os.path.join(current_dir, '..', '..','..'))
# sys.path.append(scripts_dir)
# print(sys.path[-1])
from Scripts.EPSC_Simulation.EPSC_Simulation import EPSC_Calc, Agonist_Pulse
import pandas as pd
import matplotlib.pyplot as plt
from Scripts.NSFA_Tools.EPSC_Matrix_Generator import matrix_generator
from Scripts.EPSC_Webapp.EPSC_App_Connection import multi_pool_analysis
import Scripts.NSFA_Tools.EPSC_Graphic_Utilities as egg
import numpy as np
import Scripts.EPSC_Webapp.EPSC_App_Connection as EPSC_App_Connection
import Scripts.EPSC_Webapp.EPSC_preprocessing as EPSC_preprocessing
from pandas import json_normalize

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
    NSFA_Size_Requirements_CDF_Fit()