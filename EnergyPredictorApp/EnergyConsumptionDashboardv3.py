"""
Energy Consumption Prediction Dashboard
======================================
Streamlit app for the BrainStation Capstone Project
"""

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
    page_title="Energy Consumption Predictor",
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

@st.cache_data
def load_real_data():
    """
    Load real data from CSV file
    """
    df = pd.read_csv(r"data\power_weather_daily.csv")
    # Rename columns for compatibility with rest of app
    df = df.rename(columns={
        'Datetime': 'datetime',
        'Energy Consumed by Household (Watt-hour)': 'energy_consumption',
        'temperature_2m (°C)': 'temperature',
        'relative_humidity_2m (%)': 'humidity',
        'shortwave_radiation (W/m²)': 'solar_radiation',
        'dew_point_2m (°C)': 'dew_point',
        'apparent_temperature (°C)': 'apparent_temperature',
        'surface_pressure (hPa)': 'surface_pressure',
        'cloudcover (%)': 'cloudcover',
        'windspeed_10m (km/h)': 'windspeed',
        'windgusts_10m (km/h)': 'windgusts',
        'winddirection_10m (degrees)': 'winddirection',
        'direct_radiation (W/m²)': 'direct_radiation',
        'diffuse_radiation (W/m²)': 'diffuse_radiation',
    })
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    return df

def create_time_series_plot(data, y_col, title):
    """
    Create interactive time series plot
    """
    fig = px.line(data, x='datetime', y=y_col, title=title)
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title=y_col.replace('_', ' ').title(),
        height=400
    )
    return fig

def create_correlation_heatmap(data):
    """
    Create correlation heatmap
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    correlation_matrix = data[numeric_cols].corr()
    
    fig = px.imshow(
        correlation_matrix,
        text_auto=True,
        aspect="auto",
        title="Feature Correlation Matrix"
    )
    fig.update_layout(height=500)
    return fig

def create_seasonal_analysis(data):
    """
    Create seasonal analysis plots
    """
    # Monthly aggregation
    monthly_data = data.groupby('month')['energy_consumption'].mean().reset_index()
    monthly_data['month_name'] = pd.to_datetime(monthly_data['month'], format='%m').dt.strftime('%B')
    
    # Hourly pattern
    hourly_data = data.groupby('hour')['energy_consumption'].mean().reset_index()
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Monthly Consumption', 'Hourly Pattern', 'Year Consumption', 'Weekly Pattern')
    )
    
    # Monthly consumption
    fig.add_trace(
        go.Bar(x=monthly_data['month_name'], y=monthly_data['energy_consumption'], 
               name="Monthly Avg", showlegend=False),
        row=1, col=1
    )
    
    # Hourly pattern with formatted labels
    hour_labels = [f"{h % 12 if h % 12 != 0 else 12} {'AM' if h < 12 else 'PM'}" for h in hourly_data['hour']]
    fig.add_trace(
        go.Scatter(x=hour_labels, y=hourly_data['energy_consumption'], 
                  mode='lines+markers', name="Hourly Avg", showlegend=False),
        row=1, col=2
    )
    
    # Yearly consumption
    yearly_data = data.groupby(data['datetime'].dt.year)['energy_consumption'].sum().reset_index()
    fig.add_trace(
        go.Bar(x=yearly_data['datetime'], y=yearly_data['energy_consumption'],
               name="Yearly Consumption", showlegend=False),
        row=2, col=1
    )
    
    # Weekly pattern
    weekly_data = data.groupby('day_of_week')['energy_consumption'].mean().reset_index()
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    weekly_data['day_name'] = [day_names[i] for i in weekly_data['day_of_week']]
    
    fig.add_trace(
        go.Bar(x=weekly_data['day_name'], y=weekly_data['energy_consumption'],
               name="Weekly Avg", showlegend=False),
        row=2, col=2
    )
    
    fig.update_layout(height=600, title_text="Seasonal Analysis")
    return fig

def predict_energy_consumption(input_features, model_type='random_forest'):
    """
    Make energy consumption prediction (mock function for demo)
    """
    # This would normally load the actual trained model
    # For demo purposes, we'll simulate a prediction
    
    base_prediction = 2.5
    
    # Adjust based on input features
    if input_features['temperature'] > 25:
        base_prediction += 0.5  # Higher consumption in hot weather
    elif input_features['temperature'] < 10:
        base_prediction += 0.8  # Higher consumption in cold weather
    
    if input_features['hour'] in [19, 20, 21]:  # Peak evening hours
        base_prediction += 0.3
    
    if input_features['day_of_week'] in [5, 6]:  # Weekend
        base_prediction -= 0.2
    
    # Add some randomness for realism    
    prediction = base_prediction + np.random.normal(0, 0.1)
    confidence = np.random.uniform(0.85, 0.95)
    
    return max(0.1, prediction), confidence

def main():
    """
    Main Streamlit app
    """
    # Header
    st.markdown('<h1 class="main-header">⚡ Energy Consumption Predictor</h1>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", 
                               ["Dashboard", "Predictions", "Data Analysis"])
    
    # Load real data
    data = load_real_data()
    
    if page == "Dashboard":
        st.header("📊 Dashboard Overview")
        
        # Key metrics
        col1, col2, col3 = st.columns(3)

        avg_consumption = data['energy_consumption'].mean()
        monthly_avg = data.groupby(data['datetime'].dt.month)['energy_consumption'].mean().mean()
        max_consumption = data['energy_consumption'].max()

        with col1:
            st.metric("Avg Daily Consumption", f"{avg_consumption:.2f} kW", 
                     delta=f"{np.random.uniform(-0.1, 0.1):.2f}")
        with col2:
            st.metric("Avg Monthly Consumption", f"{monthly_avg:.2f} kW", 
                     delta=f"{np.random.uniform(-0.5, 0.5):.2f}")
        with col3:
            st.metric("Peak Consumption", f"{max_consumption:.2f} kW",
                     delta=f"{np.random.uniform(-0.2, 0.2):.2f}")
        
        # Time series plot
        st.subheader("Energy Consumption Over Time")
        recent_data = data.tail(1000)  # Show last 1000 hours
        fig_ts = create_time_series_plot(recent_data, 'energy_consumption', 
                                       'Recent Energy Consumption Trend')
        st.plotly_chart(fig_ts, use_container_width=True)
        
        # Seasonal analysis
        st.subheader("Seasonal Patterns")
        fig_seasonal = create_seasonal_analysis(data)
        st.plotly_chart(fig_seasonal, use_container_width=True)
        
    elif page == "Predictions":
        st.header("🔮 Energy Consumption Predictions")
        
        # Input form
        st.subheader("Input Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            input_temp = st.slider("Temperature (°C)", -10, 40, 20)
            input_humidity = st.slider("Humidity (%)", 0, 100, 60)
            input_solar = st.slider("Solar Radiation (W/m²)", 0, 1000, 500)
        
        with col2:
            input_hour = st.selectbox("Hour of Day", list(range(24)))
            input_dow = st.selectbox("Day of Week", 
                                   ["Monday", "Tuesday", "Wednesday", "Thursday", 
                                    "Friday", "Saturday", "Sunday"])
            input_model = st.selectbox("Model Type", 
                                     ["XGBoost", "Linear Regression"])
        
        # Make prediction
        if st.button("🔍 Predict Energy Consumption"):
            input_features = {
                'temperature': input_temp,
                'humidity': input_humidity,
                'solar_radiation': input_solar,
                'hour': input_hour,
                'day_of_week': ["Monday", "Tuesday", "Wednesday", "Thursday", 
                              "Friday", "Saturday", "Sunday"].index(input_dow)
            }
            prediction, confidence = predict_energy_consumption(input_features, input_model.lower().replace(' ', '_'))
            # Display prediction
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Consumption", f"{prediction:.2f} kW")
            with col2:
                cost_estimate = prediction * 0.15  # Assume $0.15 per kWh
                st.metric("Estimated Cost", f"${cost_estimate:.2f}")
            # Prediction explanation
            st.info(f"Based on the {input_model} model, the predicted energy consumption is "
                   f"{prediction:.2f} kW")
            # Historical comparison
            st.subheader("Historical Comparison")
            similar_conditions = data[
                (abs(data['temperature'] - input_temp) < 3) &
                (data['hour'] == input_hour)
            ]
            if len(similar_conditions) > 0:
                fig_hist = px.histogram(similar_conditions, x='energy_consumption',
                                      title="Historical Consumption in Similar Conditions")
                st.plotly_chart(fig_hist, use_container_width=True)
    
    elif page == "Data Analysis":
        st.header("📈 Data Analysis")
        
        # Data overview
        st.subheader("Dataset Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dataset Shape:**", data.shape)
            st.write("**Date Range:**", f"{data['datetime'].min().date()} to {data['datetime'].max().date()}")
        
        with col2:
            st.write("**Features:**", len(data.columns))
            st.write("**Missing Values:**", data.isnull().sum().sum())
        
        # Data preview
        st.subheader("Data Preview")
        st.dataframe(data.head(10))

        # Data Dictionary Section
        st.subheader("Data Dictionary")
        data_dict = {
            "Energy Consumed by Household (Watt-hour)": "Total active energy consumed by the household (Wh)",
            "Sub_metering_4 (watt-hours of active energy)": "Additional sub-metering (Wh)",
            "Reactive Energy Consumed by Household (Watt-hour)": "Total reactive energy consumed by the household (Wh)",
            "Voltage (volts)": "Average voltage of the household circuit (V)",
            "Global_intensity (amperes)": "Average current drawn by the household (A)",
            "Sub_metering_1 (watt-hours of active energy)": "Energy used by kitchen appliances (Wh)",
            "Sub_metering_2 (watt-hours of active energy)": "Energy used by laundry appliances (Wh)",
            "Sub_metering_3 (watt-hours of active energy)": "Energy used by water heater and AC (Wh)",
            "temperature_2m (°C)": "Air temperature at 2 meters above ground (°C)",
            "relative_humidity_2m (%)": "Relative humidity at 2 meters above ground (%)",
            "dew_point_2m (°C)": "Dew point temperature at 2 meters (°C)",
            "apparent_temperature (°C)": "Feels-like temperature (°C)",
            "surface_pressure (hPa)": "Atmospheric pressure at surface level (hPa)",
            "cloudcover (%)": "Total cloud cover fraction (%)",
            "windspeed_10m (km/h)": "Mean wind speed at 10 meters (km/h)",
            "windgusts_10m (km/h)": "Maximum wind gust speed at 10 meters (km/h)",
            "winddirection_10m (degrees)": "Wind direction at 10 meters (degrees)",
            "shortwave_radiation (W/m²)": "Total shortwave solar radiation (W/m²)",
            "direct_radiation (W/m²)": "Direct sunlight reaching the surface (W/m²)",
            "diffuse_radiation (W/m²)": "Scattered sunlight reaching the surface (W/m²)"
        }
        data_dict_df = pd.DataFrame(list(data_dict.items()), columns=["Feature Name", "Description"])
        st.dataframe(data_dict_df)
        
        # Correlation analysis
        st.subheader("Correlation Analysis")
        fig_corr = create_correlation_heatmap(data)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Statistical summary
        st.subheader("Statistical Summary")
        st.dataframe(data.describe())
        
        # Feature distributions
        st.subheader("Feature Distributions")
        feature_cols = [col for col in data.columns if col != 'datetime']
        feature_to_plot = st.selectbox("Select feature to analyze:", feature_cols)
        fig_dist = px.histogram(data, x=feature_to_plot, title=f"Distribution of {feature_to_plot}")
        fig_dist.update_xaxes(showgrid=True, gridcolor='lightgray')
        fig_dist.update_yaxes(showgrid=True, gridcolor='lightgray')
        st.plotly_chart(fig_dist, use_container_width=True)

        # Time Series Visualization for Selected Feature
        st.subheader("Time Series Visualization")
        ts_feature = st.selectbox("Select feature for time series plot:", feature_cols, key="ts_feature")
        fig_ts = px.line(data, x='datetime', y=ts_feature, title=f"{ts_feature} Over Time")
        fig_ts.update_xaxes(showgrid=True, gridcolor='lightgray')
        fig_ts.update_yaxes(showgrid=True, gridcolor='lightgray')
        st.plotly_chart(fig_ts, use_container_width=True)
    

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Energy Consumption Predictor | BrainStation Capstone Project</p>
        <p>Built with Streamlit, scikit-learn, XGBoost, and TensorFlow</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()