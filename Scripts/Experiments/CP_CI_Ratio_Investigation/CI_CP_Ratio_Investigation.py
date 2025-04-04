from Scripts.EPSC_Simulation.EPSC_Simulation import EPSC_Calc
import pandas as pd
import matplotlib.pyplot as plt
from Scripts.NSFA_Tools.EPSC_Matrix_Generator import matrix_generator
from Scripts.EPSC_Webapp.EPSC_App_Connection import multi_pool_analysis
import Scripts.NSFA_Tools.EPSC_Graphic_Utilities as egg
import os

'''
The purpose of this script is to investigate how changing the CP/CI ratio and the current ratios between the two
changes the outcome of NSFA. 

The type of NSFA used will be always be PEAK START-PEAK ALIGNED-MINIMIZE ERROR

'''


##STEP 1: Create 4 simulations matching 3x/6x multiplier of CP/CI current and 10%/20% ratio of CP/CI
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
    ##Added in 0% on 4/2/2025
    # current_params = pd.DataFrame([{"iCP_Multiplier":6,"CP_Ratio":0}])
    current_params = pd.DataFrame([{"iCP_Multiplier":3,"CP_Ratio":0}])



    EPSCs_df, all_total_channel_nums, channel_data_df = EPSC_Calc(num_EPSCs=1000,channel_params=channel_params,glutamate_params=glutamate_params,current_params=current_params)

##STEP 2: Create a combined matrix and run NSFA
def NSFA_multiplier_CP_Ratios():
    ##Create a combined matrix
    output_file = 'multipliers_ratios.xlsx'
    folder_path = "C:/Users/jawad/Downloads/Python-EPSC-NSFA-Pipeline/Scripts/Experiments/CP_CI_Ratio_Investigation/EPSC_Variations"

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
        "folder_name": "C:/Users/jawad/Downloads/Python-EPSC-NSFA-Pipeline/Scripts/Experiments/CP_CI_Ratio_Investigation/NSFA_Results"

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
        glutamate_params=glutamate_params,current_params=current_param,  file_path = f'C:/Users/jawad/Downloads/Python-EPSC-NSFA-Pipeline/Scripts/Experiments/CP_CI_Ratio_Investigation/Fixed_CP_{current_param["CP_Ratio"][0]}-Mult-{current_param["iCP_Multiplier"][0]}.xlsx')


if __name__ == "__main__":
    # create_all_simulations()
    # create_fixed_simulation()
    NSFA_multiplier_CP_Ratios()
