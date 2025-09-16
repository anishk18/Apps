import streamlit as st
import importlib
import os

def main():
    st.set_page_config(
        page_title="Streamlit Apps Collection",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🚀 Streamlit Apps Collection")
    st.markdown("---")
    
    st.markdown("""
    Welcome to the Streamlit Apps Collection! This repository contains a variety of 
    interactive web applications built with Streamlit, demonstrating different 
    capabilities and use cases.
    
    Select an app from the sidebar to get started.
    """)
    
    # App descriptions
    apps = {
        "📊 Data Visualization": {
            "file": "data_viz_app",
            "description": "Interactive data visualization with charts and graphs"
        },
        "🧮 Calculator": {
            "file": "calculator_app", 
            "description": "Advanced calculator with scientific functions"
        },
        "📝 Text Analysis": {
            "file": "text_analysis_app",
            "description": "Analyze text sentiment, word frequency, and generate word clouds"
        },
        "🖼️ Image Processing": {
            "file": "image_processing_app",
            "description": "Upload and apply various filters and transformations to images"
        }
    }
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    selected_app = st.sidebar.selectbox("Choose an App:", ["Home"] + list(apps.keys()))
    
    if selected_app == "Home":
        # Display app cards on home page
        st.markdown("## Available Applications")
        
        col1, col2 = st.columns(2)
        
        for i, (app_name, app_info) in enumerate(apps.items()):
            with col1 if i % 2 == 0 else col2:
                with st.container():
                    st.markdown(f"### {app_name}")
                    st.markdown(app_info["description"])
                    if st.button(f"Launch {app_name}", key=f"btn_{i}"):
                        st.sidebar.selectbox("Choose an App:", ["Home"] + list(apps.keys()), 
                                           index=list(apps.keys()).index(app_name) + 1, key=f"nav_{i}")
        
        st.markdown("---")
        st.markdown("### 🛠️ Technologies Used")
        st.markdown("""
        - **Streamlit**: For building interactive web applications
        - **Pandas**: For data manipulation and analysis
        - **Plotly**: For interactive visualizations
        - **PIL**: For image processing
        - **TextBlob**: For text analysis and sentiment analysis
        - **WordCloud**: For generating word clouds
        - **NumPy & SciPy**: For numerical computations
        """)
        
    else:
        # Load and run selected app
        app_file = apps[selected_app]["file"]
        try:
            # Import and run the selected app
            module = importlib.import_module(f"apps.{app_file}")
            if hasattr(module, 'run'):
                module.run()
            else:
                st.error(f"App {app_file} doesn't have a 'run' function")
        except ImportError as e:
            st.error(f"Could not load app: {app_file}")
            st.error(f"Error: {str(e)}")
            st.info("Make sure all required dependencies are installed and the app file exists.")

if __name__ == "__main__":
    main()