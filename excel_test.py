import pandas as pd
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb
import streamlit as st
import numpy as np

def to_excel(df,sheet_name):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name=sheet_name)
    workbook = writer.book
    worksheet = writer.sheets[sheet_name]
    format1 = workbook.add_format({'num_format': '0.00'})
    worksheet.set_column('A:A', None, format1)
    writer.close()
    processed_data = output.getvalue()
    return processed_data

num_rows = 10
num_cols = 3
# Create random data
data = np.random.randn(num_rows, num_cols)
# Create a DataFrame
df = pd.DataFrame(data, columns=['Column_1', 'Column_2', 'Column_3'])
df_xlsx = to_excel(df)
st.download_button(label='📥 Download Current Result',
                                data=df_xlsx ,
                                file_name= 'df_test.xlsx')