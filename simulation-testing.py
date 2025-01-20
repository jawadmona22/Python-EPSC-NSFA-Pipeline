import numpy as np
import math
import pandas as pd
from transitions.extensions.diagrams import HierarchicalGraphMachine
import pyperclip
import matplotlib.pyplot as plt
from tqdm import tqdm
import streamlit as st
import random


# Define rate constants
rate_constants = {
    ('C0', 'C1'): 13660000 * 0.00002,
    ('C1', 'C0'): 2093 * 0.00002,
    ('C1', 'C2'): 6019000 * 0.00002,
    ('C2', 'C1'): 4719 * 0.00002,
    ('C2', 'O'): 17230 * 0.00002,
    ('O', 'C2'): 7468 * 0.00002,
    ('C1', 'C3'): 421.9 * 0.00002,
    ('C3', 'C1'): 31.15 * 0.00002,
    ('C2', 'C4'): 855.3 * 0.00002,
    ('C4', 'C2'): 46.65 * 0.00002,
    ('O', 'C5'): 3.108 * 0.00002,
    ('C5', 'O'): 0.6912 * 0.00002,
    ('C3', 'C4'): 6019000 * 0.00002,
    ('C4', 'C3'): 3486 * 0.00002,
    ('C4', 'C5'): 476.4 * 0.00002,
    ('C5', 'C4'): 842 * 0.00002
}

import numpy as np
from transitions.extensions import HierarchicalGraphMachine


class StateMachine:
    lotbl = -3000 * np.log(np.arange(1, 10001) * 0.0001)
    def __init__(self):
        # Define the states
        self.states = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'O']
        # Initialize the state machine
        self.machine = HierarchicalGraphMachine(model=self, states=self.states, initial='C0',show_conditions=False,title="Mermaid", graph_engine="mermaid", auto_transitions=False)
        # Define transitions
        self.machine.add_transition(trigger='transition_to_C1', source='C0', dest='C1')
        self.machine.add_transition(trigger='transition_to_C0', source='C1', dest='C0')
        self.machine.add_transition(trigger='transition_to_C2', source='C1', dest='C2')
        self.machine.add_transition(trigger='transition_to_C1', source='C2', dest='C1')
        self.machine.add_transition(trigger='transition_to_O', source='C2', dest='O')
        self.machine.add_transition(trigger='transition_to_C2', source='O', dest='C2')
        self.machine.add_transition(trigger='transition_to_C3', source='C1', dest='C3')
        self.machine.add_transition(trigger='transition_to_C1', source='C3', dest='C1')
        self.machine.add_transition(trigger='transition_to_C4', source='C2', dest='C4')
        self.machine.add_transition(trigger='transition_to_C2', source='C4', dest='C2')
        self.machine.add_transition(trigger='transition_to_C5', source='O', dest='C5')
        self.machine.add_transition(trigger='transition_to_O', source='C5', dest='O')
        self.machine.add_transition(trigger='transition_to_C4', source='C3', dest='C4')
        self.machine.add_transition(trigger='transition_to_C3', source='C4', dest='C3')
        self.machine.add_transition(trigger='transition_to_C5', source='C4', dest='C5')
        self.machine.add_transition(trigger='transition_to_C4', source='C5', dest='C4')

    def transition(self,agPulse): #agpulse is the value of the agonist pulse at the given index

        global current_rate_constant
        # Determine all possible transitions from the current state
        possible_transitions = []
        if self.state == 'C0' and agPulse == 0:
            return 0
        for (state_from, state_to), rate in rate_constants.items():
            if state_from == self.state:
                # Some transitions are dependent on the agonist, some aren't
                if (state_from == 'C1' and state_to == 'C2') or (state_from == 'C3'and state_to == 'C4'):
                    if agPulse != 0:
                        d_rateconstant = 1 / (3000 * rate *agPulse) * self.lotbl[int(10000 * random.uniform(0, 1))]
                    else:
                        d_rateconstant = 900000
                else:
                    d_rateconstant = 1 / (3000 * rate) * self.lotbl[int(10000*random.uniform(0,1))]
                possible_transitions.append((state_to, d_rateconstant))

        # Find the transition with the lowest d value
        best_transition = min(possible_transitions, key=lambda x: x[1])
        next_state = best_transition[0]
        current_rate_constant = best_transition[1]
        transition_trigger = f"transition_to_{next_state}"

        # Trigger the transition to the state with the highest d value
        getattr(self, transition_trigger)()
        return d_rateconstant



def mermaid_code_print():
    mermaid_code = state_machine.get_graph().draw(None)  # using pyperclip for convenience
    print("Graph copied to clipboard!")

    array = np.loadtxt('Simulated_EPSCs_CPCI.txt')

    # Get the shape of the array
    shape = array.shape

    print(f"Shape of the array: {shape}")
    # Convert the NumPy array to a DataFrame
    df = pd.DataFrame(array)

    # Save the DataFrame to an Excel file
    df.to_excel('sim-data.xlsx', index=False, header=False)






def Agonist_Pulse():
    agonist_pulse_array = np.zeros((2000))
    for i in range(10,1001):
        t = (i-9) * .00002
        agonist_pulse_array[i] = (5 * (1 - math.exp(-t / 0.000000002))) * math.exp(-t / 0.00002) #5x10^8 rising phase, 5x1o^4 decay phase
        # agonist_pulse_array[i] = 1000 * agonist_pulse_array[i]
        if agonist_pulse_array[i] < .001:
            agonist_pulse_array[i] = 0
    agScale =  random.uniform(1, 2) #np.random.uniform(0, 1, 2000) + 1
    agPulseScaled = 1000 * agScale * agonist_pulse_array
    return agPulseScaled




####Begin Main Call

def main():
    #UI
    # st.title("EPSC Simulator")
    agonist_pulse_array = Agonist_Pulse()
    time = np.linspace(0, 16, 800)
    plt.plot(time,agonist_pulse_array[:800]/1000)
    print(agonist_pulse_array[:23]/1000)
    plt.xlabel("Time (ms)")
    plt.ylabel("[glutamate] mM")
    plt.title("Agonist Pulse")

    plt.show()
    state_machine = StateMachine()
    iSC = 1.6
    summed_traces = []

    # Generate 200 summed traces

    for summed_iteration in tqdm(range(200), desc="Generating Summed Traces"):  # Add tqdm for progress bar
        channel_traces = []

        for channel in range(200):  # Generate 200 channel traces
            single_channel_trace = []
            for i in range(800): #That each have 800 time points
                if i <= 5:  #Initial leading fives
                    single_channel_trace.append(0)
                else:
                    if state_machine.state == "C0" and agonist_pulse_array[i] == 0:
                        single_channel_trace.append(0)
                        dXY = state_machine.transition(agonist_pulse_array[i])
                    else:
                        dXY = state_machine.transition(agonist_pulse_array[i])
                        time_in_state = math.ceil(0.5 + dXY)
                        for _ in range(time_in_state):
                            if state_machine.state == "O":
                                single_channel_trace.append(iSC)
                            else:
                                single_channel_trace.append(0)
                            i += 1
                            if i >= 800:  # Prevent exceeding the trace length
                                break

            single_channel_trace = single_channel_trace[:800]
            channel_traces.append(single_channel_trace)

        # Sum all 200 traces to get a single summed trace
        summed_trace = [sum(values) for values in zip(*channel_traces)]
        summed_traces.append(summed_trace)

    # Plot the 200 summed traces with different colors
    plt.figure(figsize=(12, 8))
    time = np.linspace(0, 16, 800)

    for trace in summed_traces:
        plt.plot(time,trace, alpha=0.5)  # Use alpha for better visibility of overlapping lines
    # plt.plot(agonist_pulse_array*(1/10))
    plt.title("EPSC Trial Simulation")
    plt.xlabel("Time (ms)")
    plt.ylabel("Current (pA)")
    plt.show()


    # Save the summed traces to an Excel file
    df = pd.DataFrame(summed_traces)
    df.T.to_excel('summed-traces-data-1.xlsx', index=False, header=False) #transpose and save





if __name__ == "__main__":
    main()