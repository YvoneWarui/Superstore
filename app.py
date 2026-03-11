import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

# Set page configuration
st.set_page_config(page_title="Superstore Interactive Web App", layout="wide")

# Cache data loading
@st.cache_data
def load_data():
    # Load the cleaned dataset (which is a CSV despite the .xls extension as verified)
    df = pd.read_csv("Superstore_Cleaned.xls", encoding='latin1')
    
    # Preprocess date columns
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Ship Date'] = pd.to_datetime(df['Ship Date'])
    
    # Feature engineering (from notebook)
    df['Year'] = df['Order Date'].dt.year
    df['Month'] = df['Order Date'].dt.month_name()
    df['Month_Num'] = df['Order Date'].dt.month
    df['Profit Margin (%)'] = (df['Profit'] / df['Sales']) * 100
    
    return df

with st.spinner("Loading Superstore Data..."):
    df = load_data()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Analytics", "Forecast"])

if page == "Dashboard":
    st.title("Superstore Dashboard")
    st.markdown("Overview of key performance indicators and sales distributions.")
    
    # Filters
    st.sidebar.subheader("Filters")
    selected_year = st.sidebar.multiselect("Select Year", options=sorted(df['Year'].unique()), default=sorted(df['Year'].unique()))
    selected_region = st.sidebar.multiselect("Select Region", options=df['Region'].unique(), default=df['Region'].unique())
    
    filtered_df = df[(df['Year'].isin(selected_year)) & (df['Region'].isin(selected_region))]
    
    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Sales", f"${filtered_df['Sales'].sum():,.2f}")
    with col2:
        st.metric("Total Profit", f"${filtered_df['Profit'].sum():,.2f}")
    with col3:
        avg_margin = filtered_df['Profit Margin (%)'].mean()
        st.metric("Avg Profit Margin", f"{avg_margin:.2f}%" if pd.notnull(avg_margin) else "0.00%")
    with col4:
        st.metric("Total Orders", f"{filtered_df['Order ID'].nunique()}")
        
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        # Sales by Category
        category_sales = filtered_df.groupby('Category')['Sales'].sum().reset_index()
        fig_cat = px.bar(category_sales, x='Category', y='Sales', title="Sales by Category", color='Category')
        st.plotly_chart(fig_cat, use_container_width=True)
        
    with col2:
        # Sales by Region
        region_sales = filtered_df.groupby('Region')['Sales'].sum().reset_index()
        fig_reg = px.pie(region_sales, values='Sales', names='Region', title="Sales by Region", hole=0.4)
        st.plotly_chart(fig_reg, use_container_width=True)
        
    # Segment Analysis
    st.subheader("Sales by Segment")
    segment_sales = filtered_df.groupby(['Segment', 'Category'])['Sales'].sum().reset_index()
    fig_seg = px.bar(segment_sales, x='Segment', y='Sales', color='Category', barmode='group', title="Segment vs Category Sales")
    st.plotly_chart(fig_seg, use_container_width=True)

elif page == "Analytics":
    st.title("Sales Analytics")
    st.markdown("Deep dive into trends and product performance.")

    st.subheader("Monthly Sales Trend")
    # Group by Month_Num and Month
    monthly_sales = df.groupby(['Month_Num', 'Month'])['Sales'].sum().reset_index()
    
    fig_monthly = px.line(monthly_sales, x='Month', y='Sales', markers=True, title="Monthly Sales Trend: When is Peak Season?",
                          labels={'Sales': 'Total Sales', 'Month': 'Month'})
    fig_monthly.update_traces(line_color='green', line_width=3, marker=dict(size=8))
    st.plotly_chart(fig_monthly, use_container_width=True)
    
    st.markdown("""
    **Observations:**
    - **August to September** shows a massive jump in sales, primarily for Office Supplies and Technology, likely correlating with Back to School season and B2B annual orders.
    - **November and December** show strong spikes, especially in Technology and Furniture due to Holiday and Black Friday impacts.
    """)
    
    st.markdown("---")
    
    st.subheader("Top 10 Best-Selling Products")
    top_products = df.groupby('Product Name')['Sales'].sum().reset_index().sort_values(by='Sales', ascending=False).head(10)
    fig_top_prod = px.bar(top_products, x='Sales', y='Product Name', orientation='h', title="Top 10 Products by Revenue",
                          color='Sales', color_continuous_scale='Blues')
    fig_top_prod.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_top_prod, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Profitability Analysis")
    yearly_summary = df.groupby('Year').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Profit Margin (%)': 'mean'
    }).reset_index()
    
    fig_profit = px.line(yearly_summary, x='Year', y='Profit Margin (%)', markers=True, 
                         title='Is the Store Getting More Efficient Each Year?',
                         hover_data=['Sales', 'Profit'])
    fig_profit.update_xaxes(type='category')
    fig_profit.update_traces(line_color='orange', line_width=3, marker=dict(size=8))
    st.plotly_chart(fig_profit, use_container_width=True)

elif page == "Forecast":
    st.title("Sales Forecast")
    st.markdown("Machine learning forecast for future sales performance.")
    
    # Prepare data for time series forecast
    # We will aggregate by month and year
    df['YearMonth'] = df['Order Date'].dt.to_period('M')
    ts_data = df.groupby('YearMonth')['Sales'].sum().reset_index()
    ts_data['YearMonth_Str'] = ts_data['YearMonth'].astype(str)
    
    # Convert 'YearMonth' to a numeric value representing an index of months since start
    ts_data['MonthIndex'] = np.arange(len(ts_data))
    
    X = ts_data[['MonthIndex']]
    y = ts_data['Sales']
    
    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict past values to show fit
    ts_data['Trend'] = model.predict(X)
    
    # Forecast future N months
    forecast_months = st.slider("Select number of months to forecast", min_value=1, max_value=24, value=6)
    
    last_index = ts_data['MonthIndex'].max()
    future_indices = pd.DataFrame({'MonthIndex': np.arange(last_index + 1, last_index + 1 + forecast_months)})
    future_sales = model.predict(future_indices)
    
    # Generate future dates
    last_date = ts_data['YearMonth'].max()
    future_dates = [last_date + i for i in range(1, forecast_months + 1)]
    future_dates_str = [str(fd) for fd in future_dates]
    
    forecast_df = pd.DataFrame({
        'YearMonth_Str': future_dates_str,
        'Sales': [None] * forecast_months,
        'Trend': future_sales,
        'Type': ['Forecast'] * forecast_months
    })
    
    ts_data['Type'] = 'Historical'
    
    combined_df = pd.concat([ts_data[['YearMonth_Str', 'Sales', 'Trend', 'Type']], forecast_df])
    
    fig_forecast = px.line(combined_df, x='YearMonth_Str', y='Sales', markers=True, title="Sales Trend and Forecast")
    fig_forecast.add_scatter(x=combined_df['YearMonth_Str'], y=combined_df['Trend'], mode='lines', name='Trend / Forecast', line=dict(dash='dash', color='red'))
    
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    st.info("The forecast uses a Linear Regression model trained on historical monthly sales to project future revenue. The red dashed line represents the overall expected trend.")
