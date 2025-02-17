import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
from openpyxl import load_workbook
from openpyxl.drawing.image import Image

# Sample dictionary data
data_dict = {
    'Column1': [1, 2, 3, 4],
    'Column2': [10, 20, 25, 30]
}

# Function to create a plot
def create_plot():
    fig, ax = plt.subplots()
    ax.plot(data_dict['Column1'], data_dict['Column2'])
    ax.set_title('Simple Plot')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    return fig

# Streamlit UI
st.title('Plot and Data to Excel Example')

# Create and display the plot
fig = create_plot()
st.pyplot(fig)

# Initialize the Excel stream
excel_stream = BytesIO()

# First, save the DataFrame to Excel
with pd.ExcelWriter(excel_stream, engine='openpyxl') as writer:
    df = pd.DataFrame(data_dict)
    df.to_excel(writer, sheet_name='Sheet1', startrow=11, startcol=0)
    writer.save()

# Load the workbook from the BytesIO stream
excel_stream.seek(0)  # Reset stream position
workbook = load_workbook(excel_stream)

# Save the plot to a BytesIO object
image_stream = BytesIO()
fig.savefig(image_stream, format='png')
image_stream.seek(0)

# Add the plot image to the workbook
ws = workbook['Sheet1']
img = Image(image_stream)
ws.add_image(img, 'G12')

# Save the updated workbook to the BytesIO stream
final_excel_stream = BytesIO()
workbook.save(final_excel_stream)
final_excel_stream.seek(0)

# Provide a download button for the Excel file
st.download_button(
    label='Download Excel File',
    data=final_excel_stream,
    file_name='data_with_plot.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)
