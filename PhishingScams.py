# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 23:19:50 2025

@author: Grace Maria IIT
"""

import pandas as pd
import plotly.express as px

# Load the phishing scam dataset
df = pd.read_csv("DownloadedDatasets\PhishingScams\phishing_scam_reports_2020_2024.csv")

# Extract year from 'Date (Year-Month)' column
df['Year'] = pd.to_datetime(df['Date (Year-Month)'], format='%Y-%m').dt.year

# Group by Year and Phishing Method to count number of reports
grouped = df.groupby(['Year', 'Phishing Method']).size().reset_index(name='Count')

# Create a stacked bar chart
fig = px.bar(
    grouped,
    x='Year',
    y='Count',
    color='Phishing Method',
    title='Phishing Scam Reports by Year and Method',
    labels={'Count': 'Number of Reports'},
    barmode='stack'
)

# Save the plot
fig.write_image("AnalysisOutputs/phishing_bar_chart.png")
# fig.write_json("AnalysisOutputs/phishing_bar_chart.json")