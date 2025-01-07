import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import io
import base64
from PIL import Image
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Advanced Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .reportview-container {
        background: linear-gradient(to right, #f8f9fa, #e9ecef)
    }
    .main {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
        width: 100%;
    }
    .stSelectbox {
        background-color: #ffffff;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stAlert {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 5px;
        padding: 10px;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .grid-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
        padding: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Header with logo and title
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.image("https://your-logo-url.png", use_column_width=True)
    st.title("🎯 Advanced Analytics Dashboard")
    st.markdown("---")

# Data Loading Functions
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

# Sidebar Configuration
st.sidebar.header("📊 Data Input")
upload_option = st.sidebar.radio("Choose input method:", ["Upload File", "API Connection"])

if upload_option == "Upload File":
    uploaded_file = st.sidebar.file_uploader("Upload your data file", type=['csv', 'xlsx', 'xls'])
    if uploaded_file:
        df = load_data_from_file(uploaded_file)
else:
    api_url = st.sidebar.text_input("Enter API URL")
    if api_url:
        df = load_data_from_api(api_url)

# Main Analysis Section
if 'df' in locals() and df is not None:
    # Quick Stats Cards
    st.header("📈 Quick Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Features", len(df.columns))
    with col3:
        st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
    with col4:
        st.metric("Data Types", len(df.dtypes.unique()))

    # Enhanced Data Overview
    st.header("🔍 Data Overview")
    
    # Advanced Data Profiling
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    date_cols = df.select_dtypes(include=['datetime64']).columns

    tab1, tab2, tab3 = st.tabs(["📊 Statistical Summary", "📋 Data Sample", "🔍 Column Info"])
    
    with tab1:
        if len(numeric_cols) > 0:
            stats_df = df[numeric_cols].describe()
            st.dataframe(stats_df.style.highlight_max(axis=0))
    
    with tab2:
        st.dataframe(df.head(10))
    
    with tab3:
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes,
            'Missing': df.isnull().sum(),
            'Missing %': (df.isnull().sum() / len(df) * 100).round(2),
            'Unique Values': df.nunique()
        })
        st.dataframe(col_info)

    # Advanced Visualizations Section
    st.header("📊 Advanced Visualizations")
    
    # 1. Time Series Analysis with Altair
    if len(date_cols) > 0:
        st.subheader("📅 Time Series Analysis")
        date_col = st.selectbox("Select Date Column", date_cols)
        metric_col = st.selectbox("Select Metric", numeric_cols)
        
        # Create time series chart with Altair
        chart = alt.Chart(df).mark_line(point=True).encode(
            x=alt.X(date_col, title='Date'),
            y=alt.Y(metric_col, title='Value'),
            tooltip=[date_col, metric_col]
        ).properties(
            width=800,
            height=400
        ).interactive()
        
        st.altair_chart(chart, use_container_width=True)
        
        # Add rolling averages
        window_size = st.slider("Select Rolling Average Window", 2, 30, 7)
        df[f'{metric_col}_rolling_avg'] = df[metric_col].rolling(window=window_size).mean()
        
        rolling_chart = alt.Chart(df).mark_line(
            color='red',
            strokeDash=[5,5]
        ).encode(
            x=date_col,
            y=f'{metric_col}_rolling_avg',
            tooltip=[date_col, f'{metric_col}_rolling_avg']
        ).properties(
            width=800,
            height=400
        )
        
        st.altair_chart(chart + rolling_chart, use_container_width=True)

    # 2. Enhanced Distribution Analysis
    st.subheader("📉 Distribution Analysis")
    dist_col = st.selectbox("Select Column for Distribution", numeric_cols)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # KDE Plot
    sns.kdeplot(data=df[dist_col], ax=ax1, fill=True)
    ax1.set_title('Kernel Density Estimation')
    
    # Enhanced Box Plot
    sns.boxenplot(y=df[dist_col], ax=ax2)
    ax2.set_title('Enhanced Box Plot')
    
    st.pyplot(fig)
    
    # 3. Interactive Correlation Matrix
    st.subheader("🔄 Correlation Analysis")
    
    corr = df[numeric_cols].corr()
    
    # Create heatmap using Altair
    corr_df = corr.reset_index().melt('index')
    corr_df.columns = ['var1', 'var2', 'correlation']
    
    base = alt.Chart(corr_df).encode(
        x='var1:O',
        y='var2:O'
    )
    
    heatmap = base.mark_rect().encode(
        color=alt.Color('correlation:Q', scale=alt.Scale(scheme='viridis'))
    )
    
    text = base.mark_text().encode(
        text=alt.Text('correlation:Q', format='.2f'),
        color=alt.condition(
            alt.datum.correlation > 0.5,
            alt.value('white'),
            alt.value('black')
        )
    )
    
    st.altair_chart(heatmap + text, use_container_width=True)

    # 4. Advanced Categorical Analysis
    if len(categorical_cols) > 0:
        st.subheader("📊 Categorical Analysis")
        cat_col = st.selectbox("Select Categorical Column", categorical_cols)
        
        # Create bar chart with Altair
        cat_chart = alt.Chart(df).mark_bar().encode(
            x=alt.X(f'{cat_col}:N', sort='-y'),
            y='count()',
            color=alt.Color(f'{cat_col}:N', scale=alt.Scale(scheme='category20')),
            tooltip=[cat_col, 'count()']
        ).properties(
            width=800,
            height=400
        ).interactive()
        
        st.altair_chart(cat_chart, use_container_width=True)

    # 5. Advanced Scatter Plot Matrix
    st.subheader("📊 Scatter Plot Matrix")
    selected_cols = st.multiselect("Select Columns for Scatter Matrix", numeric_cols)
    if len(selected_cols) >= 2:
        fig = sns.pairplot(df[selected_cols], diag_kind='kde')
        st.pyplot(fig)

    # 6. Enhanced Group Analysis
    st.subheader("👥 Group Analysis")
    if len(categorical_cols) > 0 and len(numeric_cols) > 0:
        group_col = st.selectbox("Group by", categorical_cols, key='group_analysis')
        metric_col_group = st.selectbox("Select Metric", numeric_cols, key='group_metric')
        
        group_chart = alt.Chart(df).mark_bar().encode(
            x=group_col,
            y=alt.Y(f'sum({metric_col_group}):Q', stack='normalize'),
            color=alt.Color(f'{group_col}:N', scale=alt.Scale(scheme='category20')),
            tooltip=[group_col, alt.Tooltip(f'sum({metric_col_group}):Q', format='.2f')]
        ).properties(
            width=800,
            height=400
        ).interactive()
        
        st.altair_chart(group_chart, use_container_width=True)

    # 7. Enhanced Clustering Analysis
    st.subheader("🎯 Clustering Analysis")
    cluster_cols = st.multiselect("Select Features for Clustering", numeric_cols, key='clustering')
    if len(cluster_cols) >= 2:
        n_clusters = st.slider("Number of Clusters", 2, 10, 3, key='n_clusters')
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[cluster_cols])
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['Cluster'] = kmeans.fit_predict(scaled_data)
        
        # Create scatter plot with clusters
        cluster_chart = alt.Chart(df).mark_circle(size=60).encode(
            x=cluster_cols[0],
            y=cluster_cols[1],
            color='Cluster:N',
            tooltip=cluster_cols
        ).properties(
            width=800,
            height=400
        ).interactive()
        
        st.altair_chart(cluster_chart, use_container_width=True)

    # 8. Anomaly Detection
    st.subheader("🔍 Anomaly Detection")
    anomaly_col = st.selectbox("Select Column for Anomaly Detection", numeric_cols)
    
    # Calculate z-scores
    z_scores = np.abs((df[anomaly_col] - df[anomaly_col].mean()) / df[anomaly_col].std())
    df['is_anomaly'] = z_scores > 3
    
    anomaly_chart = alt.Chart(df).mark_circle().encode(
        x=alt.X(anomaly_col, scale=alt.Scale(zero=False)),
        y=alt.Y('index', scale=alt.Scale(zero=False)),
        color='is_anomaly:N',
        size=alt.condition(alt.datum.is_anomaly, alt.value(100), alt.value(30)),
        tooltip=[anomaly_col, 'is_anomaly']
    ).properties(
        width=800,
        height=400
    ).interactive()
    
    st.altair_chart(anomaly_chart, use_container_width=True)

    # 9. Data Export Options
    st.header("📥 Export Options")
    
    # Export processed data
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Export Processed Data (CSV)"):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="processed_data.csv">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    with col2:
        if st.button("Export Processed Data (Excel)"):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='Data', index=False)
            b64 = base64.b64encode(output.getvalue()).decode()
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="processed_data.xlsx">Download Excel File</a>'
            st.markdown(href, unsafe_allow_html=True)

    # 10. Advanced Settings
    st.sidebar.header("⚙️ Advanced Settings")
    
    # Theme Selection
    theme = st.sidebar.selectbox(
        "Select Theme",
        ["Light", "Dark", "Custom"],
        key='theme_selection'
    )
    
    if theme == "Custom":
        primary_color = st.sidebar.color_picker("Primary Color", "#4CAF50")
        st.markdown(f"""
        <style>
        .stButton>button {{
            background-color: {primary_color} !important;
        }}
        </style>
        """, unsafe_allow_html=True)

    # Additional Analysis Options
    st.sidebar.header("📊 Additional Analysis")
