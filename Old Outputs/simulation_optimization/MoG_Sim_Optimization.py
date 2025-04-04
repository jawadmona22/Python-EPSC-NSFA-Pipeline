import numpy as np
import pandas as pd
import EPSC_Graphic_Utilities as egg
import EPSC_Simulation as simulator
from tqdm import tqdm

base_file_name = 'C:/Users/jawad/Downloads/Python-EPSC-NSFA-Pipeline/'
def sample_mog_params():
    """Randomly sample parameters for a Mixture of Gaussians."""
    num_components = np.random.choice([1, 2, 3])  # Randomly pick 1 to 3 Gaussians

    means = np.random.uniform(200, 4000, size=num_components)  # Sample means
    std_devs = np.random.uniform(50, 800, size=num_components)  # Sample std devs
    weights = np.random.dirichlet(np.ones(num_components), size=1)[0]  # Normalize weights

    return {"means": means, "std_devs": std_devs, "weights": weights}

def sample_normal_params():
    """Randomly sample parameters for a normal distribution."""

    mean = np.random.uniform(200, 4000)  # Sample means
    std_dev = np.random.uniform(20, 800)  # Sample std devs

    return {"mean": mean, "std_dev": std_dev}


def generate_training_data(num_samples=100):
    """Generate training data using Mixture of Gaussians and EPSC_Calc."""
    training_data = []

    for _ in tqdm(range(num_samples), desc="Generating Training Data"):
        #mog_params = sample_mog_params()

        normal_parms = sample_normal_params()
        # Sample glutamate scale
        glutamate_scale = int(np.random.uniform(1, 10))

        # Run the simulation
        output_df, total_num_channels = simulator.EPSC_Calc(channel_distribution="normal",glutamate_scale=glutamate_scale,mean=normal_parms["mean"], sd = normal_parms["std_dev"])

        output_df = output_df.T
        folder_name = "C:/Users/jawad/Downloads/Python-EPSC-NSFA-Pipeline/simulation_optimization/"
        # Extract summary features
        peak_amplitudes = np.array(egg.amplitude_histogram_creator(output_df,folder_name = folder_name))
        rise_times = egg.rise_times_histogram_creator(output_df,folder_name = folder_name)
        decay_taus = egg.tau_graph_generator(output_df,folder_name = folder_name)

        # Store results
        # training_data.append({
        #     "num_components": len(mog_params["means"]),
        #     "means": mog_params["means"],
        #     "std_devs": mog_params["std_devs"],
        #     "weights": mog_params["weights"],
        #     "glutamate_scale": glutamate_scale,
        #     "num_channels": total_num_channels,
        #     "peak_amplitudes": peak_amplitudes,
        #     "rise_times": rise_times,
        #     "decay_taus": decay_taus
        # })
        training_data.append({
            "mean": normal_parms["mean"],
            "std_dev": normal_parms["std_dev"],
            "std_dev": normal_parms["std_dev"],
            "glutamate_scale": glutamate_scale,
            "num_channels": total_num_channels,
            "peak_amplitudes": peak_amplitudes,
            "rise_times": rise_times,
            "decay_taus": decay_taus
        })
    return pd.DataFrame(training_data)


# Generate dataset
training_df = generate_training_data(num_samples=100)
training_df.to_pickle(f"{base_file_name}simulation_optimization/training_df_v6_100normal.pkl")

