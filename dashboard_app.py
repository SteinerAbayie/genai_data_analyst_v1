import streamlit as st
import pandas as pd
import plotly.express as px

# Load data
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('result.csv')
        return data
    except FileNotFoundError:
        st.error("Data file not found. Please ensure 'result.csv' is present.")
        return pd.DataFrame()

data = load_data()

# Set up the page configuration
st.set_page_config(
    page_title="California City Churn Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dark theme
st.markdown(
    """
    <style>
    body {
        background-color: #1E1E1E;
        color: #FFFFFF;
        font-family: 'Inter', sans-serif;
    }
    .css-18e3th9 {
        background-color: #2D2D2D;
    }
    .css-1d391kg {
        background-color: #2D2D2D;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and subtitle
st.title("California City Churn Dashboard")
st.subheader("Visualizing churn amounts across cities in California")

# Filters
cities = data['City'].unique() if not data.empty else []
selected_cities = st.sidebar.multiselect("Select Cities", options=cities, default=cities)

# Filter data based on selection
filtered_data = data[data['City'].isin(selected_cities)] if not data.empty else pd.DataFrame()

# KPIs
if not filtered_data.empty:
    total_churn_count = filtered_data['churn_count'].sum()
else:
    total_churn_count = 0

st.metric(label="Total Churn Count", value=f"{total_churn_count:,}", help="Total number of churns across all cities")

# Charts
if not filtered_data.empty:
    # Heatmap
    heatmap_fig = px.density_heatmap(
        filtered_data,
        x='City',
        y='churn_count',
        z='churn_count',
        histfunc='sum',
        color_continuous_scale=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8'],
        title="Churn Heatmap by City"
    )
    heatmap_fig.update_layout(
        template="plotly_dark",
        font=dict(family="Inter, sans-serif", color="#FFFFFF"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        title=dict(font=dict(size=20, color="#FFFFFF")),
        margin=dict(t=60, b=50, l=50, r=50),
        hoverlabel=dict(bgcolor="white", font_size=12)
    )
    st.plotly_chart(heatmap_fig, use_container_width=True)

    # Histogram
    histogram_fig = px.histogram(
        filtered_data,
        x='churn_count',
        title="Distribution of churn_count",
        color_discrete_sequence=['#FF6B6B']
    )
    histogram_fig.update_layout(
        template="plotly_dark",
        font=dict(family="Inter, sans-serif", color="#FFFFFF"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        title=dict(font=dict(size=20, color="#FFFFFF")),
        margin=dict(t=60, b=50, l=50, r=50),
        hoverlabel=dict(bgcolor="white", font_size=12)
    )
    st.plotly_chart(histogram_fig, use_container_width=True)
else:
    st.warning("No data available for the selected filters.")