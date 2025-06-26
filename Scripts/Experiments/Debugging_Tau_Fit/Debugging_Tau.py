import Scripts.NSFA_Tools.EPSC_Graphic_Utilities as egg
import matplotlib.pyplot as plt
import pandas as pd



def debug_tau_histogram():
    EPSCs = pd.read_excel(r"C:\Users\jawad\Downloads\Python-EPSC-NSFA-Pipeline\Scripts\Experiments\Debugging_Tau_Fit\Decay Taus and traces_JG21O03A.xlsx",sheet_name=1)
    print(EPSCs.shape)
    taus = egg.tau_graph_generator(EPSCs,folder_name="",debug=False)
    taus_df = pd.DataFrame(taus)
    taus_df.to_excel("Taus_Debugged.xlsx")


if __name__ == "__main__":
    debug_tau_histogram()