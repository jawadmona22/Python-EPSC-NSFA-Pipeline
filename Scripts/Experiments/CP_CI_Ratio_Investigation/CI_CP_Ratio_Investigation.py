import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.abspath(os.path.join(current_dir, '..', '..','..'))
sys.path.append(scripts_dir)
print(sys.path[-1])
from Scripts.EPSC_Simulation.EPSC_Simulation import EPSC_Calc, Agonist_Pulse
import pandas as pd
import matplotlib.pyplot as plt
from Scripts.NSFA_Tools.EPSC_Matrix_Generator import matrix_generator
from Scripts.EPSC_Webapp.EPSC_App_Connection import multi_pool_analysis
import Scripts.NSFA_Tools.EPSC_Graphic_Utilities as egg
import numpy as np

'''
The purpose of this script is to investigate how changing the CP\\CI ratio and the current ratios between the two
changes the outcome of NSFA. 

The type of NSFA used will be always be PEAK START-PEAK ALIGNED-MINIMIZE ERROR

'''
os.chdir('C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\Scripts\\Experiments\\CP_CI_Ratio_Investigation')


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
    file_name = 'C:\\Users\\j.mona\\Documents\\GitHub\\Python-EPSC-NSFA-Pipeline\\data-files\\Many_EPSCs_Juan.xlsx'
    folder_name = 'C:\\Users\\j.mona\\Documents\\GitHub\\Python-EPSC-NSFA-Pipeline\\Scripts\\Experiments\\CP_CI_Ratio_Investigation\\Median_CDF_Results\\'
    egg.generate_meanECDF_data(file_name=file_name,folder_name=folder_name,use_mean =True, use_median=True,plt_show=True)

def get_global_median_tau():
    file_name = 'C:\\Users\\j.mona\\Documents\\GitHub\\Python-EPSC-NSFA-Pipeline\\data-files\\Many_EPSCs_Juan.xlsx'
    folder_name = 'C:\\Users\\j.mona\\Documents\\GitHub\\Python-EPSC-NSFA-Pipeline\\Scripts\\Experiments\\CP_CI_Ratio_Investigation\\Median_CDF_Results\\'

    all_taus = []
    for name, sheet in pd.read_excel(file_name, sheet_name=None).items():
        print(sheet.shape,name)
        taus_array = egg.tau_graph_generator(sheet,folder_name,False)
        all_taus.append(taus_array)
    all_taus = pd.DataFrame(all_taus)
    all_taus.to_excel('C:\\Users\\j.mona\\Documents\\GitHub\\Python-EPSC-NSFA-Pipeline\\Scripts\\Experiments\\CP_CI_Ratio_Investigation\\Median_CDF_Results\\all_taus.xlsx')
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
        EPSCs_df, all_total_channel_nums, channel_data_df = EPSC_Calc(num_EPSCs=10,channel_params=channel_params,
        glutamate_params=glutamate_params,current_params=current_param,  output_file_path = f'C:\\Users\\j.mona\\Documents\\GitHub\\Python-EPSC-NSFA-Pipeline\\Scripts\\Experiments\\CP_CI_Ratio_Investigation\\EPSCS_Steady_Agonist_Fixed_n\\R-{current_param["CP_Ratio"][0]}-Mult-{current_param["iCP_Multiplier"][0]}.xlsx',
        folder_path='C:\\Users\\j.mona\\Documents\\GitHub\\Python-EPSC-NSFA-Pipeline\\Scripts\\Experiments\\CP_CI_Ratio_Investigation\\EPSCS_Steady_Agonist_Fixed_n')


def NSFA_steady_agonist():
     ##Create a combined matrix
    output_file = 'steady_EPSCs.xlsx'
    folder_path = "C:\\Users\\jawad\\Downloads\\Python-EPSC-NSFA-Pipeline\\Scripts\\Experiments\\CP_CI_Ratio_Investigation\\EPSCS_Steady_Agonist_Fixed_n"

    with pd.ExcelWriter(output_file) as writer:
            for file in os.listdir(f'{os.getcwd()}/EPSCS_Steady_Agonist_Fixed_n'):
                if file.endswith(".xlsx"):
                    file_path = os.path.join(os.getcwd(),file)
                    df = pd.read_excel(file_path)
                    # Write the file to an Excel sheet
                    #Remove xlsx
                    df.to_excel(writer, sheet_name=str(file)[7:], index=False, header=False) #7: is to make the name <=31 chars

    params = {
        "alignment": ["peak"],
        "analysis_start_point": ["peak_start"],
        "scaling": ["minimize_error"],
        "output": ["linear", "parabolic"],
        "file_name": output_file,
        "folder_name": output_folder

    }

    matrix = matrix_generator(params,first_sheet=False)
    matrix_df = pd.DataFrame(matrix)
    matrix_df.to_excel('steady_agonists_matrix.xlsx')       

if __name__ == "__main__":
    # print(f"Directory {sys.path}")
    # create_all_simulations()
    # create_fixed_simulation()
    NSFA_multiplier_CP_Ratios()
    # create_median_tau_CDFs()
    # get_global_median_tau()
    # EPSCs_steady_agonist()
    # NSFA_steady_agonist()


