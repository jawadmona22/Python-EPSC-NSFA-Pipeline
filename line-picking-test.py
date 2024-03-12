# import streamlit as st
# import plotly.express as px
# import numpy as np
#
# # Generate some sample data
# x = np.linspace(0, 10, 100)
# y1 = np.sin(x)
# y2 = np.cos(x)
#
# # Create a DataFrame
# df = px.data.iris()
#
# # Create a line plot using Plotly Express
# fig = px.line(x=x, y=[y1, y2], labels={'variable': 'Function', 'value': 'Value'})
#
# # Define callback for selection event
# def highlight_selected_line(trace, points, selector):
#     if points:
#         selected_trace_index = points.trace_index
#         for i, _ in enumerate(fig.data):
#             if i == selected_trace_index:
#                 fig.data[i].line.color = 'gold'
#             else:
#                 fig.data[i].line.color = 'blue'  # Change 'blue' to whatever original color you have
#     else:
#         for i, _ in enumerate(fig.data):
#             fig.data[i].line.color = 'blue'  # Change 'blue' to whatever original color you have
#
# # Register callback for selection event
# fig.data[0].on_selection(highlight_selected_line)
#
# # Display the figure in Streamlit
# st.plotly_chart(fig)
import streamlit as st
import plotly.express as px
import pandas as pd
from streamlit_plotly_events import plotly_events

x = [1, 2, 3, 4, 5]
y = [6, 7, 2, 4, 5]

df=[]
df= pd.DataFrame(df)
df['year']= x
df['lifeExp']= y

fig = px.line(df, x="year", y="lifeExp", title='Life expectancy in Canada')

selected_points = plotly_events(fig)
a=selected_points[0]
a= pd.DataFrame.from_dict(a,orient='index')
a
