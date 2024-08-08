import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots





def process_cont_data(file_content,num_recordings):
    lines = file_content.decode('utf-8').split('\n')

    # Remove any empty lines
    lines = [line.strip() for line in lines if line.strip()]

    # Extract the values from the second column and convert to float
    values = [float(line.split()[1]) for line in lines]

    # Determine the number of samples per recording
    num_samples_per_recording = len(values) // num_recordings

    # Reshape the values into a 2D array with 5 columns
    data_array = np.array(values).reshape(-1, num_samples_per_recording).T
    data_array = data_array[0:10000,:]
    return data_array


def main():
    st.title('Electrical Recordings Data Processor')

    # File uploader
    uploaded_file = st.file_uploader("Upload text file", type=['txt'])

    if uploaded_file is not None:
        # Process the uploaded file
        file_content = uploaded_file.read()
        data_array = process_cont_data(file_content,5)

        # Display the resulting NumPy array
        st.subheader('Processed Data:')
        st.write(data_array)
        fig, axs = plt.subplots(5, 1)
        num_plots = data_array.shape[1]

        # Display the figure using Streamlit
        fig = make_subplots(rows=6, cols=1, shared_xaxes=False)

        for i in range(num_plots):
            data = data_array[2000:9999,i]

            # Add line plots of the data to each subplot
            print(i)
            fig.add_trace(go.Scatter(x=np.arange(len(data)), y=data, mode='lines'), row=i+1, col=1)

            # Set the layout
            fig.update_layout(title='Recordings before processing', xaxis_title='Time (ms)')

            # Display the figure using Streamlit
        st.plotly_chart(fig)



if __name__ == "__main__":
    main()
