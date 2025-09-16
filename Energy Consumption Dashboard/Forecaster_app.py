"""
Energy Consumption  Dashboard
======================================
Streamlit app for the BrainStation Capstone Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import warnings
from PIL import Image
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Energy Consumption DashBoard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

# Main Title
st.markdown('<div class="main-header">⚡ Energy Consumption Dashboard ⚡</div>', unsafe_allow_html=True)

# Section 1: Data Loading
st.header("1. Data Loading")
st.markdown("Load and preview the monthly energy and weather dataset.")

def load_data(path):
    df = pd.read_csv(path)
    return df

df = load_data(r"C:\Users\karni\OneDrive\Documents\GitHub\Apps\Energy Consumption Dashboard\data\power_weather_monthly.csv")
st.dataframe(df)


# Section 2: Time Series Visualization
st.header("Time Series Visualization")
st.markdown("Visualize the monthly averaged features over time trend.")

# Dropdown to select variable for time series plot
time_series_variable = st.selectbox(
    "Choose a Dataset Feature to plot against time:",
    [col for col in df.columns if col != 'DateTime']
)

fig_selected = px.line(df, x='DateTime', 
                       y=time_series_variable, 
                       title=f'{time_series_variable} Over Time')

st.plotly_chart(fig_selected, use_container_width=True)





# Section 3: Feature Visualizations
st.header(" Feature Distribution")

# Dropdown to select feature (exclude DateTime)
feature_cols = [col for col in df.columns if col != 'DateTime']
selected_feature = st.selectbox(
    "Select a feature for visualization:",
    feature_cols
)

# Distribution
st.subheader(f"Distribution of {selected_feature}")
fig_hist = px.histogram(
    df,
    x=selected_feature,
    nbins=30,
    title=f'Distribution of {selected_feature}',
    color_discrete_sequence=['#1f77b4']
)
fig_hist.update_layout(
    bargap=0.15,
    plot_bgcolor='black',
    paper_bgcolor='black',
    font=dict(size=16, color='white'),
    title_x=0.5,
    margin=dict(l=40, r=40, t=60, b=40),
    xaxis=dict(
        title=selected_feature,
        showgrid=True,
        showticklabels=True,
        tickfont=dict(size=14, color='white'),
        titlefont=dict(size=16, color='white'),
        automargin=True,
        linecolor='white',
        gridcolor='gray'
    ),
    yaxis=dict(
        title="Count",
        showgrid=True,
        showticklabels=True,
        tickfont=dict(size=14, color='white'),
        titlefont=dict(size=16, color='white'),
        automargin=True,
        linecolor='white',
        gridcolor='gray'
    )
)
fig_hist.update_traces(marker_line_width=1, marker_line_color='white')
st.plotly_chart(fig_hist, use_container_width=True)


#Energy Consumption vs Selected Feature (Scatter Plot)
st.subheader(" Scatter Plot of Energy Consumption vs DataSet Features")

# Dropdown to select x-axis variable for scatter plot (exclude DateTime and target)
scatter_x_options = [col for col in df.columns if col not in ['DateTime', 'Energy Consumed by Household (Watt-hour)']]
selected_scatter_x = st.selectbox(
    "Select feature for plotting against Energy Consumption:",
    scatter_x_options
)

fig_scatter = px.scatter(
    df,
    x=selected_scatter_x,
    y='Energy Consumed by Household (Watt-hour)',
    title=f'Energy Consumption vs {selected_scatter_x}',
    color_discrete_sequence=['#ff7f0e']
)
fig_scatter.update_layout(
    xaxis_title=selected_scatter_x,
    yaxis_title='Energy Consumed by Household (Watt-hour)',
    xaxis=dict(
        showgrid=True,
        showticklabels=True,
        tickfont=dict(size=14),
        titlefont=dict(size=16),
        automargin=True
    ),
    yaxis=dict(
        showgrid=True,
        showticklabels=True,
        tickfont=dict(size=14),
        titlefont=dict(size=16),
        automargin=True
    ),
    font=dict(size=16),
    title_x=0.5,
    margin=dict(l=40, r=40, t=60, b=40)
)
st.plotly_chart(fig_scatter, use_container_width=True)




# Section 4: Yearly Data View
st.header("Energy Consumption by Year")

# Extract year from DateTime column
df['Year'] = pd.to_datetime(df['DateTime']).dt.year

# Dropdown to select year
years = sorted(df['Year'].unique())
selected_year = st.selectbox("Select Year:", years)

# Filter data for selected year
year_df = df[df['Year'] == selected_year]


# Time series visualization for selected year
fig_yearly = px.line(
    year_df,
    x='DateTime',
    y='Energy Consumed by Household (Watt-hour)',
    title=f"Energy Consumed by Household (Watt-hour) in {selected_year}",
    color_discrete_sequence=['#2ca02c']
)
fig_yearly.update_layout(title_x=0.5)
st.plotly_chart(fig_yearly, use_container_width=True)



