# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from io import BytesIO
import base64
from PIL import Image
import requests
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Advanced Data Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #f5f5f5;
}
.stApp {
    background-image: url("https://your-background-image-url.jpg");
    background-size: cover;
}
.logo-img {
    max-width: 200px;
}
</style>
""", unsafe_allow_html=True)

# Header Section
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.image("https://your-logo-url.png", use_column_width=True)
    st.title("📊 Advanced Data Analysis Dashboard")

# File Upload Section
st.sidebar.header("Data Input")
upload_option = st.sidebar.radio("Choose input method:", ["Upload File", "API Connection"])

@st.cache_data
def load_data_from_file(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file)
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

@st.cache_data
def load_data_from_api(api_url):
    try:
        response = requests.get(api_url)
        data = response.json()
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

# Data Loading
if upload_option == "Upload File":
    uploaded_file = st.sidebar.file_uploader("Upload your data file", type=['csv', 'xlsx', 'xls'])
    if uploaded_file:
        df = load_data_from_file(uploaded_file)
else:
    api_url = st.sidebar.text_input("Enter API URL")
    if api_url:
        df = load_data_from_api(api_url)

if 'df' in locals() and df is not None:
    # Quick Insights Section
    st.header("📈 Quick Insights")
    
    # Basic Dataset Information
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Total Features", len(df.columns))
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    with col4:
        st.metric("Duplicate Rows", df.duplicated().sum())
    
    # Data Overview
    st.subheader("Key Insights")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 0:
        summary_stats = df[numeric_cols].agg(['mean', 'median', 'std', 'min', 'max']).round(2)
        st.write("Statistical Summary:", summary_stats)
    
    # Visualization Section
    st.header("📊 Data Visualizations")
    
    # 1. Time Series Analysis (if date column exists)
    date_cols = df.select_dtypes(include=['datetime64']).columns
    if len(date_cols) > 0:
        st.subheader("Time Series Analysis")
        date_col = st.selectbox("Select Date Column", date_cols)
        metric_col = st.selectbox("Select Metric", numeric_cols)
        
        # Time Series Plot
        fig = px.line(df, x=date_col, y=metric_col, title=f"{metric_col} Over Time")
        st.plotly_chart(fig)
        
        # Seasonal Decomposition
        try:
            decomposition = seasonal_decompose(df[metric_col], period=12)
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12))
            decomposition.observed.plot(ax=ax1)
            ax1.set_title('Observed')
            decomposition.trend.plot(ax=ax2)
            ax2.set_title('Trend')
            decomposition.seasonal.plot(ax=ax3)
            ax3.set_title('Seasonal')
            decomposition.resid.plot(ax=ax4)
            ax4.set_title('Residual')
            st.pyplot(fig)
        except:
            st.warning("Seasonal decomposition not possible with current data")
    
    # 2. Distribution Analysis
    st.subheader("Distribution Analysis")
    selected_col = st.selectbox("Select Column for Distribution", numeric_cols)
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}")
        st.plotly_chart(fig)
    with col2:
        fig = px.box(df, y=selected_col, title=f"Box Plot of {selected_col}")
        st.plotly_chart(fig)
    
    # 3. Correlation Analysis
    st.subheader("Correlation Analysis")
    correlation = df[numeric_cols].corr()
    fig = px.imshow(correlation, title="Correlation Heatmap")
    st.plotly_chart(fig)
    
    # 4. Categorical Analysis
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        st.subheader("Categorical Analysis")
        cat_col = st.selectbox("Select Categorical Column", categorical_cols)
        fig = px.pie(df, names=cat_col, title=f"Distribution of {cat_col}")
        st.plotly_chart(fig)
    
    # 5. Scatter Plot Matrix
    st.subheader("Scatter Plot Matrix")
    selected_cols = st.multiselect("Select Columns for Scatter Matrix", numeric_cols)
    if len(selected_cols) > 1:
        fig = px.scatter_matrix(df[selected_cols])
        st.plotly_chart(fig)
    
    # 6. Group Analysis
    st.subheader("Group Analysis")
    if len(categorical_cols) > 0 and len(numeric_cols) > 0:
        group_col = st.selectbox("Group by", categorical_cols)
        metric_col = st.selectbox("Select Metric for Group Analysis", numeric_cols)
        grouped_data = df.groupby(group_col)[metric_col].agg(['mean', 'sum', 'count'])
        st.write(grouped_data)
        
        fig = px.bar(grouped_data, y='sum', title=f"Sum of {metric_col} by {group_col}")
        st.plotly_chart(fig)
    
    # 7. Time-based Patterns
    if len(date_cols) > 0:
        st.subheader("Time-based Patterns")
        date_col = df[date_cols[0]]
        df['Year'] = date_col.dt.year
        df['Month'] = date_col.dt.month
        df['Day'] = date_col.dt.day
        df['DayOfWeek'] = date_col.dt.dayofweek
        
        metric_col = st.selectbox("Select Metric for Time Analysis", numeric_cols)
        fig = px.box(df, x='Month', y=metric_col, title=f"{metric_col} Distribution by Month")
        st.plotly_chart(fig)
    
    # 8. Clustering Analysis
    st.subheader("Clustering Analysis")
    cluster_cols = st.multiselect("Select Features for Clustering", numeric_cols)
    if len(cluster_cols) > 1:
        n_clusters = st.slider("Number of Clusters", 2, 10, 3)
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[cluster_cols])
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['Cluster'] = kmeans.fit_predict(scaled_data)
        
        # Visualize clusters
        if len(cluster_cols) >= 2:
            fig = px.scatter(df, x=cluster_cols[0], y=cluster_cols[1], 
                           color='Cluster', title="Cluster Analysis")
            st.plotly_chart(fig)
    
    # 9. Outlier Detection
    st.subheader("Outlier Detection")
    outlier_col = st.selectbox("Select Column for Outlier Detection", numeric_cols)
    q1 = df[outlier_col].quantile(0.25)
    q3 = df[outlier_col].quantile(0.75)
    iqr = q3 - q1
    outliers = df[(df[outlier_col] < (q1 - 1.5 * iqr)) | (df[outlier_col] > (q3 + 1.5 * iqr))]
    st.write(f"Number of outliers detected: {len(outliers)}")
    
    fig = px.box(df, y=outlier_col, title=f"Outliers in {outlier_col}")
    st.plotly_chart(fig)
    
    # 10. Custom Visualization Builder
    st.subheader("Custom Visualization Builder")
    chart_type = st.selectbox("Select Chart Type", 
                            ["Line", "Bar", "Scatter", "Box", "Violin", "Area"])
    x_axis = st.selectbox("X-axis", df.columns)
    y_axis = st.selectbox("Y-axis", df.columns)
    color_option = st.selectbox("Color by", ["None"] + list(df.columns))
    
    if chart_type == "Line":
        fig = px.line(df, x=x_axis, y=y_axis, color=None if color_option == "None" else color_option)
    elif chart_type == "Bar":
        fig = px.bar(df, x=x_axis, y=y_axis, color=None if color_option == "None" else color_option)
    elif chart_type == "Scatter":
        fig = px.scatter(df, x=x_axis, y=y_axis, color=None if color_option == "None" else color_option)
    elif chart_type == "Box":
        fig = px.box(df, x=x_axis, y=y_axis, color=None if color_option == "None" else color_option)
    elif chart_type == "Violin":
        fig = px.violin(df, x=x_axis, y=y_axis, color=None if color_option == "None" else color_option)
    else:  # Area
        fig = px.area(df, x=x_axis, y=y_axis, color=None if color_option == "None" else color_option)
    
    st.plotly_chart(fig)
    
    # Export Section
    st.header("📥 Export Options")
    
    # Export processed data
    if st.button("Export Processed Data"):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="processed_data.csv">Download Processed Data</a>'
        st.markdown(href, unsafe_allow_html=True)
    
    # Export visualizations
    if st.button("Export All Visualizations"):
        # Implementation for exporting all visualizations
        st.info("Feature coming soon!")
    
    # Additional Features Section
    st.header("🔧 Additional Features")
    
    # Data Cleaning Options
    st.subheader("Data Cleaning")
    if st.checkbox("Remove Duplicates"):
        df = df.drop_duplicates()
        st.success("Duplicates removed!")
    
    if st.checkbox("Handle Missing Values"):
        missing_strategy = st.selectbox("Choose strategy", 
                                      ["Drop", "Fill with Mean", "Fill with Median"])
        if missing_strategy == "Drop":
            df = df.dropna()
        elif missing_strategy == "Fill with Mean":
            df = df.fillna(df.mean())
        else:
            df = df.fillna(df.median())
        st.success("Missing values handled!")
    
    # Feature Engineering
    st.subheader("Feature Engineering")
    if len(numeric_cols) > 0:
        if st.checkbox("Add Normalized Columns"):
            for col in numeric_cols:
                df[f"{col}_normalized"] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
            st.success("Normalized columns added!")
    
    # Settings
    st.sidebar.header("Settings")
    theme = st.sidebar.selectbox("Chart Theme", ["plotly", "plotly_white", "plotly_dark"])
    
    # Help & Documentation
    st.sidebar.header("Help")
    if st.sidebar.checkbox("Show Documentation"):
        st.sidebar.markdown("""
        ### How to use this dashboard:
        1. Upload your data file or connect via API
        2. Explore various visualizations
        3. Use the custom visualization builder
        4. Export your results
        """)
else:
    st.info("Please upload a file or connect to an API to begin analysis")
