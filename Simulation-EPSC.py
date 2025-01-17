import numpy as np

# Define rate constants (scaled by time step)
kC0C1 = 0.1  # Transition from C0 to C1
kC1C0 = 0.05  # Transition from C1 to C0
kC1C2 = 0.2  # Transition from C1 to C2
kC2C1 = 0.1  # Transition from C2 to C1
kC2O = 0.3   # Transition from C2 to O
kOC2 = 0.2   # Transition from O to C2

# Define simulation parameters
time_step = 0.01  # Time step (ms)
total_time = 100  # Total simulation time (ms)
n_steps = int(total_time / time_step)  # Number of steps

# Initialize state variables
state_C0 = np.zeros(n_steps)
state_C1 = np.zeros(n_steps)
state_C2 = np.zeros(n_steps)
state_O = np.zeros(n_steps)

# Initial conditions
state_C0[0] = 1  # Start with all channels in C0

# Function to simulate state transitions
def simulate_state_transitions():
    for t in range(1, n_steps):
        Rnd = np.random.random()

        # Transition from C0
        if state_C0[t - 1] > 0:
            if Rnd < kC0C1 * time_step:
                state_C1[t] += 1
            else:
                state_C0[t] += 1

        # Transition from C1
        if state_C1[t - 1] > 0:
            if Rnd < kC1C0 * time_step:
                state_C0[t] += 1
            elif Rnd < (kC1C0 + kC1C2) * time_step:
                state_C2[t] += 1
            else:
                state_C1[t] += 1

        # Transition from C2
        if state_C2[t - 1] > 0:
            if Rnd < kC2C1 * time_step:
                state_C1[t] += 1
            elif Rnd < (kC2C1 + kC2O) * time_step:
                state_O[t] += 1
            else:
                state_C2[t] += 1

        # Transition from O
        if state_O[t - 1] > 0:
            if Rnd < kOC2 * time_step:
                state_C2[t] += 1
            else:
                state_O[t] += 1

        # Normalize to ensure total probability conservation
        total = state_C0[t] + state_C1[t] + state_C2[t] + state_O[t]
        if total > 0:
            state_C0[t] /= total
            state_C1[t] /= total
            state_C2[t] /= total
            state_O[t] /= total

# Run the simulation
simulate_state_transitions()

# Calculate EPSC (as an example, we assume current proportional to open state)
EPSC = state_O * 10  # Arbitrary scaling factor

# Plotting results (requires matplotlib)
import matplotlib.pyplot as plt

time = np.linspace(0, total_time, n_steps)
plt.plot(time, state_C0, label="C0")
plt.plot(time, state_C1, label="C1")
plt.plot(time, state_C2, label="C2")
plt.plot(time, state_O, label="O")
plt.plot(time, EPSC, label="EPSC", linestyle='--')
plt.xlabel("Time (ms)")
plt.ylabel("State Probabilities / EPSC")
plt.legend()
plt.show()
