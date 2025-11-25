# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 20:48:44 2025

@author: Grace Maria IIT
"""

import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
df = pd.read_csv("DownloadedDatasets/GlobalAIContentImpactAnalysis2020_2025/Global_AI_Content_Impact_Dataset.csv")

# Quick check
print("Columns:", df.columns.tolist())
print(df.head().to_string())

# Step 2: plots

# 1. AI Adoption Rate by Country
fig1 = px.bar(df, x="Country", y="AI Adoption Rate (%)", color="Industry",
              title="AI Adoption Rate by Country and Industry",
              barmode="group")
fig1.write_image("AnalysisOutputs/adoptionRate.png")

# 2. AI-Generated Content Volume vs Year
fig2 = px.scatter(df, x="Year", y="AI-Generated Content Volume (TBs per year)",
                  color="Country", size="AI Adoption Rate (%)",
                  title="Content Volume Over Years",
                  labels={"AI-Generated Content Volume (TBs per year)": "Content Volume (TB/year)"})
fig2.write_image("AnalysisOutputs/ContentVolumevsYear.png")

# 3. Heatmap: Correlation between numeric features
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap of AI Impact Metrics")
plt.write_image("AnalysisOutputs/CorelationHeatmap.png")

# 4. Job Loss vs Revenue Increase (Bubble Chart)
fig3 = px.scatter(df, x="Job Loss Due to AI (%)", y="Revenue Increase Due to AI (%)",
                  size="Human-AI Collaboration Rate (%)", color="Industry",
                  hover_name="Country", title="Job Loss vs Revenue Increase")
fig3.write_image("AnalysisOutputs/JobLossvsRevenue.png")


# 5. Consumer Trust vs Market Share by Regulation Status
fig4 = px.box(df, x="Regulation Status", y="Consumer Trust in AI (%)",
              color="Regulation Status", title="Consumer Trust by Regulation Status")
fig4.write_image("AnalysisOutputs/TrustvsStatus.png")
