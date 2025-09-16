import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

def run():
    st.title("📊 Data Visualization App")
    st.markdown("---")
    
    # Sidebar for options
    st.sidebar.header("📋 Data Options")
    
    # Data source selection
    data_source = st.sidebar.selectbox(
        "Choose Data Source:",
        ["Sample Dataset", "Upload CSV", "Generate Random Data"]
    )
    
    df = None
    
    if data_source == "Sample Dataset":
        # Create sample dataset
        np.random.seed(42)
        n_points = 1000
        df = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=n_points, freq='D'),
            'Sales': np.random.normal(1000, 200, n_points) + np.sin(np.arange(n_points) * 0.1) * 100,
            'Category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], n_points),
            'Region': np.random.choice(['North', 'South', 'East', 'West'], n_points),
            'Profit': np.random.normal(150, 50, n_points)
        })
        df['Sales'] = np.maximum(df['Sales'], 0)  # Ensure non-negative values
        
    elif data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.sidebar.success(f"✅ Loaded {len(df)} rows")
            except Exception as e:
                st.sidebar.error(f"❌ Error loading file: {e}")
        else:
            st.info("👆 Please upload a CSV file to proceed")
            return
            
    elif data_source == "Generate Random Data":
        n_points = st.sidebar.slider("Number of data points:", 100, 5000, 1000)
        np.random.seed(42)
        df = pd.DataFrame({
            'X': np.random.normal(0, 1, n_points),
            'Y': np.random.normal(0, 1, n_points),
            'Z': np.random.exponential(2, n_points),
            'Category': np.random.choice(['A', 'B', 'C', 'D'], n_points)
        })
    
    if df is not None:
        # Display basic info
        st.subheader("📋 Dataset Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        
        # Show data preview
        if st.checkbox("📖 Show Data Preview"):
            st.dataframe(df.head(10), use_container_width=True)
        
        # Statistical summary
        if st.checkbox("📈 Show Statistical Summary"):
            st.subheader("Statistical Summary")
            st.dataframe(df.describe(), use_container_width=True)
        
        st.markdown("---")
        
        # Visualization options
        st.subheader("📊 Visualizations")
        
        # Select columns for visualization
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        datetime_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        if numeric_columns:
            # Chart type selection
            chart_type = st.selectbox(
                "Select Chart Type:",
                ["Line Chart", "Bar Chart", "Scatter Plot", "Histogram", "Box Plot", "Correlation Heatmap"]
            )
            
            if chart_type == "Line Chart":
                if datetime_columns and len(datetime_columns) > 0:
                    x_col = st.selectbox("X-axis (Date):", datetime_columns)
                    y_col = st.selectbox("Y-axis:", numeric_columns)
                    
                    fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} over Time")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No datetime columns found for line chart")
            
            elif chart_type == "Bar Chart":
                if categorical_columns:
                    x_col = st.selectbox("X-axis (Category):", categorical_columns)
                    y_col = st.selectbox("Y-axis (Value):", numeric_columns)
                    
                    chart_data = df.groupby(x_col)[y_col].mean().reset_index()
                    fig = px.bar(chart_data, x=x_col, y=y_col, title=f"Average {y_col} by {x_col}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No categorical columns found for bar chart")
            
            elif chart_type == "Scatter Plot":
                if len(numeric_columns) >= 2:
                    x_col = st.selectbox("X-axis:", numeric_columns)
                    y_col = st.selectbox("Y-axis:", [col for col in numeric_columns if col != x_col])
                    color_col = st.selectbox("Color by (optional):", ["None"] + categorical_columns)
                    
                    if color_col == "None":
                        fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
                    else:
                        fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} vs {x_col}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Need at least 2 numeric columns for scatter plot")
            
            elif chart_type == "Histogram":
                col = st.selectbox("Select Column:", numeric_columns)
                bins = st.slider("Number of Bins:", 10, 100, 30)
                
                fig = px.histogram(df, x=col, nbins=bins, title=f"Distribution of {col}")
                st.plotly_chart(fig, use_container_width=True)
            
            elif chart_type == "Box Plot":
                y_col = st.selectbox("Y-axis (Numeric):", numeric_columns)
                x_col = st.selectbox("X-axis (Category, optional):", ["None"] + categorical_columns)
                
                if x_col == "None":
                    fig = px.box(df, y=y_col, title=f"Box Plot of {y_col}")
                else:
                    fig = px.box(df, x=x_col, y=y_col, title=f"Box Plot of {y_col} by {x_col}")
                st.plotly_chart(fig, use_container_width=True)
            
            elif chart_type == "Correlation Heatmap":
                if len(numeric_columns) >= 2:
                    correlation_matrix = df[numeric_columns].corr()
                    
                    fig = px.imshow(
                        correlation_matrix,
                        labels=dict(color="Correlation"),
                        title="Correlation Heatmap",
                        color_continuous_scale="RdBu"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Need at least 2 numeric columns for correlation heatmap")
        else:
            st.warning("No numeric columns found in the dataset for visualization")
        
        # Advanced analytics
        st.markdown("---")
        st.subheader("🔍 Advanced Analytics")
        
        if st.checkbox("📊 Generate Insights"):
            st.write("**Dataset Insights:**")
            
            # Missing values
            missing_values = df.isnull().sum()
            if missing_values.sum() > 0:
                st.write(f"• Missing values found in {(missing_values > 0).sum()} columns")
            else:
                st.write("• No missing values detected ✅")
            
            # Data types
            st.write(f"• Data types: {df.dtypes.value_counts().to_dict()}")
            
            # Unique values
            for col in categorical_columns[:3]:  # Show only first 3 categorical columns
                unique_count = df[col].nunique()
                st.write(f"• '{col}' has {unique_count} unique values")

if __name__ == "__main__":
    run()