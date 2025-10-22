# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 20:08:07 2025

@author: Grace Maria IIT
"""

import pandas as pd
import plotly.express as px

# Step 1: Load dataset with correct headers
columns = [
    "ID", "Title", "Category", "Attack Type", "Scenario Description", "Tools Used",
    "Attack Steps", "Target Type", "Vulnerability", "MITRE Technique", "Impact",
    "Detection Method", "Solution", "Tags", "Source"
]
df = pd.read_csv("DownloadedDatasets/CyberAttacksGlobal/Attack_Dataset.csv", names=columns, skiprows=1)

# Step 2: Group data by Attack Type and Target Type
grouped = df.groupby(['Attack Type', 'Target Type']).size().reset_index(name='Count')

# Step 3: Filter top Attack Types and Impact categories
top_attack_types = df['Attack Type'].value_counts().head(150).index
top_impacts = df['Target Type'].value_counts().head(100).index
filtered = grouped[grouped['Attack Type'].isin(top_attack_types) & grouped['Target Type'].isin(top_impacts)]

# Step 4: Create stacked bar chart using Plotly
fig = px.bar(
    filtered,
    x='Attack Type',
    y='Target Type',
    title='Chart: Attack Type vs Target Type',
    barmode='stack', width=1536, height=1024
)

# Step 5: Show and export chart
# fig.show()
fig.write_image("AnalysisOutputs/attack_vs_target_type.png")