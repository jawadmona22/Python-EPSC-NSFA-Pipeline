from EPSC_Matrix_Generator import matrix_generator
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinterdnd2 import DND_FILES, TkinterDnD
import pandas as pd

# --- Your function here (modified to accept params and file) ---
def test_NSFA_analysis(params):
    print("TESTING NSFA ANALYSIS")
    file_name = params["file_name"]
    folder_path = params["folder_name"]

    # Example: call your matrix_generator
    matrix = matrix_generator(params, first_sheet=True, debug=True)
    matrix_df = pd.DataFrame(matrix)
    matrix_df.to_excel('testing_NSFA_matrix.xlsx')
    print("Saved: testing_NSFA_matrix.xlsx")


# --- GUI ---
class NSFAApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NSFA Analysis Tool")

        # File selection (drag-drop area)
        self.file_label = tk.Label(root, text="Drag & Drop Excel File Here", width=50, height=4, relief="ridge")
        self.file_label.pack(pady=10)
        self.file_label.drop_target_register(DND_FILES)
        self.file_label.dnd_bind('<<Drop>>', self.drop_file)

        # Parameters
        self.params = {
            "direct_df_input": None, #tk.BooleanVar(value=False),
            "alignment": tk.StringVar(value="peak"),
            "analysis_start_point": tk.StringVar(value="peak_start"),
            "scaling": tk.StringVar(value="minimize_error"),
            "output": tk.StringVar(value="linear"),
            "recording_duration": tk.IntVar(value=16),
            "file_name": None,
            "folder_name": "EPSCs_Test_Files",
            "sheet_names": [0]
        }

        # Dropdown menus
        # self.make_dropdown("Direct DF Input:", [False, True], self.params["direct_df_input"])
        self.make_dropdown("Alignment:", ["peak", "other_option"], self.params["alignment"])
        self.make_dropdown("Start Point:", ["peak_start", "baseline"], self.params["analysis_start_point"])
        self.make_dropdown("Scaling:", ["minimize_error", "normalize"], self.params["scaling"])
        self.make_checkboxes("Output Types:", ["linear", "parabolic"], "output")
        self.make_integer_input("Recording Duration (ms):", self.params["recording_duration"])

        # Run button
        run_button = tk.Button(root, text="Run Analysis", command=self.run_analysis)
        run_button.pack(pady=15)

    def make_dropdown(self, label, options, var):
        frame = tk.Frame(self.root)
        frame.pack(pady=3)
        tk.Label(frame, text=label).pack(side="left", padx=5)
        ttk.Combobox(frame, textvariable=var, values=options, state="readonly").pack(side="left")

    def make_integer_input(self, label, var, default=1000):
        frame = tk.Frame(self.root)
        frame.pack(pady=3)
        tk.Label(frame, text=label).pack(side="left", padx=5)
        entry = tk.Entry(frame, textvariable=var, width=10)
        entry.pack(side="left")
        var.set(default)  # set default value
        return entry

    def make_checkboxes(self, label, options, param_key):
        frame = tk.Frame(self.root)
        frame.pack(pady=3, anchor="w")
        tk.Label(frame, text=label).pack(anchor="w")

        # Store BooleanVars for each option
        vars_dict = {}
        for opt in options:
            var = tk.BooleanVar(value=False)
            chk = tk.Checkbutton(frame, text=opt, variable=var)
            chk.pack(anchor="w", padx=20)
            vars_dict[opt] = var

        self.params[param_key] = vars_dict

    def drop_file(self, event):
        file_path = event.data.strip("{}")  # Handle spaces in filename
        self.params["file_name"] = file_path
        self.file_label.config(text=f"Selected File:\n{file_path}")

    def run_analysis(self):
        if not self.params["file_name"]:
            messagebox.showerror("Error", "Please drag & drop an Excel file first!")
            return
        try:
            test_NSFA_analysis(self.params)
            messagebox.showinfo("Success", "Analysis complete! Output saved.")
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed:\n{e}")


if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = NSFAApp(root)
    root.mainloop()




