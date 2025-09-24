import os
import sys
from Scripts.EPSC_Simulation.EPSC_Simulation import EPSC_Calc, Agonist_Pulse
import pandas as pd
import matplotlib.pyplot as plt
from Scripts.NSFA_Tools.EPSC_Matrix_Generator import matrix_generator
from Scripts.EPSC_Webapp.EPSC_App_Connection import multi_pool_analysis
import Scripts.NSFA_Tools.EPSC_Graphic_Utilities as egg
import numpy as np

'''The purpose of this script is to create a validation for NSFA to prevent future errors and to ensure
alignment and all other variables are working properly. This is simply a validation  of the "debug" feature for the scripts
involved with EPSC Simulation-->NSFA Matrix Generator'''


def EPSC_Simulator():
    glutamate_params = pd.DataFrame(
        [{"gl_mean": None, "gl_sd": None, "distribution_type": "fixed_value", "fixed_value": 10,"continuous":False}])

    channel_params = pd.DataFrame(
        [{"distribution_type": "fixed_value", "channel_sd": None, "channel_mean": None, "fixed_value": 100}])


    current_params = pd.DataFrame([{"iCP_Multiplier": 3, "CP_Ratio": 0}]) #Essentially, only CI
    folder_path = r'C:\Users\jawad\Downloads\Python-EPSC-NSFA-Pipeline\data-files\test.xlsx'
    EPSCs_df, all_total_channel_nums, channel_data_df = EPSC_Calc(num_EPSCs=500, channel_params=channel_params,
                                                                  glutamate_params=glutamate_params,
                                                                  current_params=current_params,output_file_path=folder_path)


def test_NSFA_analysis():
    print("TESTING NSFA ANALYSIS")
    ##Create a combined matrix
    file_name = 'EPSCs_unspecified.xlsx'
    folder_path = 'EPSCs_Test_Files'

    params = {
        "direct_df_input": None,
        "alignment": ["peak"],
        "analysis_start_point": ["peak_start",],
        "scaling": ["minimize_error"],
        "output": ["linear", "parabolic"],
        "file_name": file_name,
        "folder_name": folder_path

    }

    matrix = matrix_generator(params, first_sheet=True,debug=True)
    matrix_df = pd.DataFrame(matrix)
    matrix_df.to_excel('testing_NSFA_matrix.xlsx')

def test_alignment_debug():
    print("TESTING Alignment Debug ANALYSIS")
    ##Create a combined matrix
    file_name = 'EPSCs_unspecified.xlsx'
    folder_path = 'EPSCs_Test_Files'

    params = {
        "direct_df_input": None,
        "alignment": ["max_dv_dt"],
        "analysis_start_point": ["peak_start","alignment_point"],
        "scaling": ["minimize_error"],
        "output": ["linear", "parabolic"],
        "file_name": file_name,
        "folder_name": folder_path,
        "recording_duration":16

    }

    matrix = matrix_generator(params, first_sheet=True,debug=True)
    matrix_df = pd.DataFrame(matrix)
    matrix_df.to_excel('testing_alignment_matrix.xlsx')




if __name__ == '__main__':
    EPSC_Simulator()
    # test_NSFA_analysis()
    # test_alignment_debug()