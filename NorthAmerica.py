# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 22:46:53 2025

@author: Grace Maria IIT
"""

import pandas as pd
import plotly.express as px

# Load the dataset
df = pd.read_csv("DownloadedDatasets\DataBreachesNorthAmerica\DataBreaches.csv")

# Group data by Year and Method to count number of breaches
breach_counts = df.groupby(['Year', 'Method']).size().reset_index(name='Count')

# Create a stacked bar chart
fig = px.bar(
    breach_counts,
    x='Year',
    y='Count',
    color='Method',
    title='Number of Breaches Over Time by Method',
    labels={'Count': 'Number of Breaches', 'Year': 'Year', 'Method': 'Breach Method'}
)

# Save the plot as JSON and PNG files
fig.write_json("stacked_breaches_by_year.json")
fig.write_image("stacked_breaches_by_year.png")