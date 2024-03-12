import PySimpleGUI as sg
import EPSC_Pipeline_Jan_18_24 as pipeline

#Set the theme and base column
sg.theme('Kayak')
sg.SetOptions()
instructionColumn = [
    [sg.Text(
        "Instructions: \n Select your EPSC trace file. All columns must be the same length. \n Accepted file types: xls, csv.")]]

layout = [
    [sg.T('NSFA Analysis', font=("Times New Roman", 15))],
    [instructionColumn],
    [sg.Text("File: ", font=("Times New Roman", 12)), sg.In(size=(70, 1), enable_events=True, key="-IMAGE-"),
     sg.FileBrowse(key='-FILE-')],
    [sg.B('Analyze File'), sg.B('Save Analysis'), sg.B('Exit'), sg.Text(key='Error')],
    [sg.T('Aggregate Graph: Mean Current vs. Variance:')],
    [sg.Column(layout=[[sg.Canvas(key='fig_cv', size=(800, 400))]], background_color='#DAE0E6', pad=(0, 0))],
    [sg.Text("Unitary Current: ", key="-CURRENT-")]
]


_FILECHOSEN = False
_FOLDERCHOSEN = False
_ANALYZEDFO = False
window = sg.Window(title='Graph with controls', layout=layout, auto_size_text=True, auto_size_buttons=True)
window.Finalize()

while True:
    event, values = window.Read()

    if event in [None, 'Exit']:  # always,  always give a way out!
        window.Close()
        break
