import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(
    page_title="Apps Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title
st.title("📊 Apps Dashboard")
st.markdown("Welcome to the interactive Apps Dashboard! This application provides various tools and visualizations.")

# Sidebar
st.sidebar.header("🛠️ Dashboard Controls")
app_mode = st.sidebar.selectbox(
    "Choose App Mode",
    ["Home", "Data Visualization", "Calculator", "Text Analysis", "Random Data Generator"]
)

if app_mode == "Home":
    st.header("🏠 Home")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="📈 Features Available",
            value="5",
            delta="All functional"
        )
    
    with col2:
        st.metric(
            label="🔧 Built with",
            value="Streamlit",
            delta="Latest version"
        )
    
    with col3:
        st.metric(
            label="📅 Last Updated",
            value=datetime.now().strftime("%Y-%m-%d"),
            delta="Today"
        )
    
    st.markdown("---")
    
    st.subheader("✨ Available Features")
    
    features = [
        {"name": "📊 Data Visualization", "description": "Create interactive charts and graphs"},
        {"name": "🧮 Calculator", "description": "Perform mathematical calculations"},
        {"name": "📝 Text Analysis", "description": "Analyze text content and statistics"},
        {"name": "🎲 Random Data Generator", "description": "Generate sample data for testing"},
    ]
    
    for feature in features:
        with st.expander(feature["name"]):
            st.write(feature["description"])

elif app_mode == "Data Visualization":
    st.header("📊 Data Visualization")
    
    viz_type = st.selectbox(
        "Select Visualization Type",
        ["Line Chart", "Bar Chart", "Scatter Plot", "Histogram", "Pie Chart"]
    )
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    data = pd.DataFrame({
        'Date': dates,
        'Sales': np.random.normal(1000, 200, len(dates)).cumsum(),
        'Users': np.random.poisson(100, len(dates)),
        'Revenue': np.random.exponential(500, len(dates)),
        'Category': np.random.choice(['A', 'B', 'C', 'D'], len(dates))
    })
    
    if viz_type == "Line Chart":
        st.subheader("📈 Line Chart - Sales Over Time")
        fig = px.line(data, x='Date', y='Sales', title='Sales Trend Over Time')
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "Bar Chart":
        st.subheader("📊 Bar Chart - Users by Category")
        category_data = data.groupby('Category')['Users'].sum().reset_index()
        fig = px.bar(category_data, x='Category', y='Users', title='Total Users by Category')
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "Scatter Plot":
        st.subheader("🔍 Scatter Plot - Users vs Revenue")
        fig = px.scatter(data, x='Users', y='Revenue', color='Category', title='Users vs Revenue')
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "Histogram":
        st.subheader("📊 Histogram - Revenue Distribution")
        fig = px.histogram(data, x='Revenue', bins=30, title='Revenue Distribution')
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "Pie Chart":
        st.subheader("🥧 Pie Chart - Category Distribution")
        category_counts = data['Category'].value_counts()
        fig = px.pie(values=category_counts.values, names=category_counts.index, title='Category Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    # Show raw data
    if st.checkbox("Show Raw Data"):
        st.subheader("📋 Raw Data")
        st.dataframe(data.head(10))

elif app_mode == "Calculator":
    st.header("🧮 Calculator")
    
    calc_type = st.radio(
        "Calculator Type",
        ["Basic Calculator", "Scientific Calculator", "Unit Converter"]
    )
    
    if calc_type == "Basic Calculator":
        st.subheader("➕ Basic Calculator")
        
        col1, col2 = st.columns(2)
        with col1:
            num1 = st.number_input("Enter first number:", value=0.0, format="%.2f")
        with col2:
            num2 = st.number_input("Enter second number:", value=0.0, format="%.2f")
        
        operation = st.selectbox("Select operation:", ["+", "-", "×", "÷"])
        
        if st.button("Calculate"):
            if operation == "+":
                result = num1 + num2
            elif operation == "-":
                result = num1 - num2
            elif operation == "×":
                result = num1 * num2
            elif operation == "÷":
                if num2 != 0:
                    result = num1 / num2
                else:
                    st.error("Cannot divide by zero!")
                    result = None
            
            if result is not None:
                st.success(f"Result: {num1} {operation} {num2} = {result:.2f}")
    
    elif calc_type == "Scientific Calculator":
        st.subheader("🔬 Scientific Calculator")
        
        number = st.number_input("Enter number:", value=0.0, format="%.4f")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Square Root"):
                if number >= 0:
                    st.write(f"√{number} = {np.sqrt(number):.4f}")
                else:
                    st.error("Cannot calculate square root of negative number!")
        
        with col2:
            if st.button("Square"):
                st.write(f"{number}² = {number**2:.4f}")
        
        with col3:
            if st.button("Sine"):
                st.write(f"sin({number}) = {np.sin(number):.4f}")

elif app_mode == "Text Analysis":
    st.header("📝 Text Analysis")
    
    text_input = st.text_area("Enter text to analyze:", height=200)
    
    if text_input:
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Characters", len(text_input))
        with col2:
            st.metric("Words", len(text_input.split()))
        with col3:
            st.metric("Lines", len(text_input.split('\n')))
        with col4:
            st.metric("Paragraphs", len([p for p in text_input.split('\n\n') if p.strip()]))
        
        st.markdown("---")
        
        # Word frequency
        st.subheader("📊 Word Frequency")
        words = text_input.lower().split()
        word_freq = pd.Series(words).value_counts().head(10)
        
        if len(word_freq) > 0:
            fig = px.bar(
                x=word_freq.index,
                y=word_freq.values,
                title="Top 10 Most Frequent Words"
            )
            st.plotly_chart(fig, use_container_width=True)

elif app_mode == "Random Data Generator":
    st.header("🎲 Random Data Generator")
    
    st.subheader("Generate Sample Dataset")
    
    col1, col2 = st.columns(2)
    with col1:
        num_rows = st.slider("Number of rows:", 10, 1000, 100)
    with col2:
        data_type = st.selectbox("Data type:", ["Sales Data", "User Data", "Product Data"])
    
    if st.button("Generate Data"):
        np.random.seed(None)  # Use different seed each time
        
        if data_type == "Sales Data":
            generated_data = pd.DataFrame({
                'Date': pd.date_range(start='2024-01-01', periods=num_rows, freq='D'),
                'Product': np.random.choice(['Product A', 'Product B', 'Product C', 'Product D'], num_rows),
                'Sales': np.random.normal(1000, 300, num_rows).round(2),
                'Quantity': np.random.poisson(10, num_rows),
                'Region': np.random.choice(['North', 'South', 'East', 'West'], num_rows)
            })
        elif data_type == "User Data":
            generated_data = pd.DataFrame({
                'User_ID': range(1, num_rows + 1),
                'Age': np.random.randint(18, 80, num_rows),
                'Score': np.random.normal(75, 15, num_rows).round(1),
                'Active': np.random.choice([True, False], num_rows),
                'Category': np.random.choice(['Premium', 'Standard', 'Basic'], num_rows)
            })
        else:  # Product Data
            generated_data = pd.DataFrame({
                'Product_ID': range(1, num_rows + 1),
                'Price': np.random.exponential(50, num_rows).round(2),
                'Rating': np.random.uniform(1, 5, num_rows).round(1),
                'Stock': np.random.poisson(50, num_rows),
                'Category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Sports'], num_rows)
            })
        
        st.subheader("📋 Generated Data")
        st.dataframe(generated_data)
        
        # Download button
        csv = generated_data.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"{data_type.lower().replace(' ', '_')}_{num_rows}_rows.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")