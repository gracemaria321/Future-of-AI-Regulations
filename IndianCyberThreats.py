# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 23:27:26 2025

@author: Grace Maria IIT
"""

import pandas as pd
import plotly.express as px

# Load dataset
file_path = "DownloadedDatasets\CyberThreatsIndia\cyber-crimes-from-ncrb-master-data-year-state-and-city-wise-type-of-cyber-crimes-committed-in-violation-of-it-act.csv"
df = pd.read_csv(file_path)

# Group by year and offence_category, summing the values
grouped = df.groupby(['year', 'offence_category'])['value'].sum().reset_index()

# Create a stacked bar chart
fig = px.bar(
    grouped,
    x='year',
    y='value',
    color='offence_category',
    title='Cyber Crimes in India (NCRB): Offence Categories Over Years',

    labels={'year': 'Year', 'value': 'Number of Offences', 'offence_category': 'Offence Category'},
    width=1536, height=1024
)

# Save the plot
fig.write_image("AnalysisOutputs/India_cyber_crimes.png")
# fig.write_json("AnalysisOutputs/india_cyber_crimes.json")