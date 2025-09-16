# 🚀 Streamlit Apps Collection

A comprehensive collection of interactive web applications built with Streamlit, showcasing various features and capabilities of the framework.

## 📋 Available Applications

### 1. 📊 Data Visualization App
Interactive data visualization tool with multiple chart types and data analysis features.

**Features:**
- Upload CSV files or use sample datasets
- Generate random data for testing
- Multiple chart types: Line, Bar, Scatter, Histogram, Box Plot, Correlation Heatmap
- Statistical analysis and insights
- Data preprocessing and exploration

### 2. 🧮 Advanced Calculator
Multi-purpose calculator with basic, scientific, unit conversion, and financial calculations.

**Features:**
- Basic arithmetic operations
- Scientific functions (trigonometry, logarithms, etc.)
- Unit conversions (length, weight, temperature, area, volume)
- Financial calculations (interest, loans, investments)
- Mathematical constants and functions

### 3. 📝 Text Analysis App
Comprehensive text analysis tool with sentiment analysis and text processing capabilities.

**Features:**
- Text statistics (word count, character count, etc.)
- Word frequency analysis with stop word filtering
- Word cloud generation
- Sentiment analysis (polarity and subjectivity)
- Text preprocessing and pattern detection
- Email and URL extraction

### 4. 🖼️ Image Processing App
Image manipulation and enhancement tool with various filters and effects.

**Features:**
- Image upload support (PNG, JPG, JPEG, GIF, BMP)
- Filters: Blur, Sharpen, Edge Enhancement, Emboss
- Enhancements: Brightness, Contrast, Saturation, Sharpness
- Transformations: Rotate, Flip, Resize, Crop
- Color adjustments: Grayscale, Sepia, Invert, Posterize, Solarize
- Special effects: Vintage, Color temperature, Soft focus
- Download processed images

## 🚀 Getting Started

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/anishk18/Apps.git
cd Apps
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Applications

1. Start the main dashboard:
```bash
streamlit run main.py
```

2. Open your web browser and navigate to `http://localhost:8501`

3. Select any application from the sidebar to explore its features

### Running Individual Apps

You can also run individual applications directly:

```bash
# Data Visualization App
streamlit run apps/data_viz_app.py

# Calculator App
streamlit run apps/calculator_app.py

# Text Analysis App
streamlit run apps/text_analysis_app.py

# Image Processing App
streamlit run apps/image_processing_app.py
```

## 🛠️ Technologies Used

- **Streamlit**: Web app framework for Python
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Plotly**: Interactive visualizations
- **Matplotlib & Seaborn**: Statistical data visualization
- **Pillow (PIL)**: Image processing
- **TextBlob**: Natural language processing
- **WordCloud**: Word cloud generation
- **SciPy**: Scientific computing
- **Scikit-learn**: Machine learning library

## 📁 Project Structure

```
Apps/
├── main.py                     # Main dashboard application
├── requirements.txt            # Python dependencies
├── README.md                  # Project documentation
├── .gitignore                 # Git ignore file
└── apps/                      # Individual applications
    ├── __init__.py
    ├── data_viz_app.py         # Data visualization app
    ├── calculator_app.py       # Calculator app
    ├── text_analysis_app.py    # Text analysis app
    └── image_processing_app.py # Image processing app
```

## 🤝 Contributing

Contributions are welcome! Here are some ways you can contribute:

1. **Bug Reports**: Found a bug? Please create an issue with details
2. **Feature Requests**: Have an idea for a new feature? Let us know!
3. **Code Contributions**: 
   - Fork the repository
   - Create a feature branch
   - Make your changes
   - Submit a pull request

### Adding New Apps

To add a new Streamlit app:

1. Create a new Python file in the `apps/` directory
2. Implement a `run()` function that contains your Streamlit app code
3. Add your app to the `apps` dictionary in `main.py`
4. Update the requirements.txt if you use new dependencies

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🆘 Support

If you encounter any issues or have questions:

1. Check the existing issues in the repository
2. Create a new issue with detailed information
3. Include steps to reproduce any bugs

## 🌟 Features Overview

### Data Visualization
- **Multiple Data Sources**: Upload CSV, generate random data, or use samples
- **Interactive Charts**: Plotly-powered interactive visualizations
- **Statistical Insights**: Automatic data analysis and insights generation
- **Data Export**: Download charts and processed data

### Calculator
- **Multi-mode**: Basic, Scientific, Unit Converter, Financial
- **History**: Track recent calculations
- **Comprehensive**: Support for complex mathematical operations
- **User-friendly**: Intuitive interface with clear results

### Text Analysis
- **Multi-input**: Type, upload files, or use sample texts
- **Comprehensive Analysis**: Statistics, frequency, sentiment
- **Visualization**: Word clouds and frequency charts
- **Pattern Detection**: Automatic extraction of emails, URLs, etc.

### Image Processing
- **Multiple Formats**: Support for common image formats
- **Real-time Preview**: See changes instantly
- **Batch Effects**: Apply multiple filters and enhancements
- **Download**: Save processed images in PNG format

## 🔧 Configuration

The applications use default Streamlit configurations, but you can customize them by:

1. Creating a `.streamlit/config.toml` file
2. Modifying the page configuration in individual apps
3. Adjusting the sidebar and layout settings

## 📊 Performance

- **Optimized**: Efficient processing for large datasets and images
- **Responsive**: Fast loading and interactive updates
- **Memory-aware**: Proper memory management for large files
- **Error Handling**: Comprehensive error handling and user feedback