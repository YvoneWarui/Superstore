import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

# Set page configuration
st.set_page_config(page_title="Superstore Analytics", layout="wide", page_icon="📈")

# Cache data loading
@st.cache_data
def load_data():
    df = pd.read_csv("Superstore_Cleaned.xls", encoding='latin1')
    
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Ship Date'] = pd.to_datetime(df['Ship Date'])
    
    df['Year'] = df['Order Date'].dt.year
    df['Month'] = df['Order Date'].dt.month_name()
    df['Month_Num'] = df['Order Date'].dt.month
    df['Profit Margin (%)'] = (df['Profit'] / df['Sales']) * 100
    
    return df

with st.spinner("Loading Superstore Data..."):
    df = load_data()

# Sidebar: Navigation and Project Info
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Analytics", "Forecast"])

st.sidebar.markdown("---")
st.sidebar.subheader("Global Filters")
# Global Filters for Dashboard and Analytics
selected_year = st.sidebar.multiselect("Select Year", options=sorted(df['Year'].unique()), default=sorted(df['Year'].unique()))
selected_region = st.sidebar.multiselect("Select Region", options=df['Region'].unique(), default=df['Region'].unique())

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 🛒 **Superstore Analytics**
Welcome to the interactive analytics dashboard for the Superstore dataset.

**Features:**
- **Dashboard**: High-level KPI metrics and regional performance.
- **Analytics**: Deep dive into seasonal trends and product profitability.
- **Forecast**: Predictive modeling using historical sales data.
""")

# Apply global filters to data
filtered_df = df[(df['Year'].isin(selected_year)) & (df['Region'].isin(selected_region))]

if page == "Dashboard":
    st.title("📊 Superstore Dashboard")
    st.markdown("Overview of key performance indicators and sales distributions.")
    
    if filtered_df.empty:
        st.warning("No data available for the selected filters. Please adjust your selection in the sidebar.")
    else:
        # KPI Metrics
        st.markdown("### Key Performance Indicators")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Sales", f"${filtered_df['Sales'].sum():,.2f}")
        with col2:
            st.metric("Total Profit", f"${filtered_df['Profit'].sum():,.2f}")
        with col3:
            avg_margin = filtered_df['Profit Margin (%)'].mean()
            st.metric("Avg Profit Margin", f"{avg_margin:.2f}%" if pd.notnull(avg_margin) else "0.00%")
        with col4:
            st.metric("Total Orders", f"{filtered_df['Order ID'].nunique():,}")
            
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            # Sales by Category
            category_sales = filtered_df.groupby('Category')['Sales'].sum().reset_index()
            fig_cat = px.bar(
                category_sales, 
                x='Category', 
                y='Sales', 
                title="Total Sales by Category", 
                color='Category',
                text_auto='.2s'
            )
            fig_cat.update_layout(showlegend=False)
            st.plotly_chart(fig_cat, width="stretch")
            
        with col2:
            # Sales by Region
            region_sales = filtered_df.groupby('Region')['Sales'].sum().reset_index()
            fig_reg = px.pie(
                region_sales, 
                values='Sales', 
                names='Region', 
                title="Sales Distribution by Region", 
                hole=0.4
            )
            fig_reg.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_reg, width="stretch")
            
        # Segment Analysis
        st.markdown("### Sales by Segment & Category")
        segment_sales = filtered_df.groupby(['Segment', 'Category'])['Sales'].sum().reset_index()
        fig_seg = px.bar(
            segment_sales, 
            x='Segment', 
            y='Sales', 
            color='Category', 
            barmode='group', 
            title="Segment vs Category Output"
        )
        st.plotly_chart(fig_seg, width="stretch")

elif page == "Analytics":
    st.title("📈 Sales Analytics")
    st.markdown("Deep dive into trends, seasonality, and product performance based on selected filters.")
    
    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
    else:
        st.subheader("Monthly Sales Trend")
        monthly_sales = filtered_df.groupby(['Month_Num', 'Month'])['Sales'].sum().reset_index()
        
        fig_monthly = px.line(
            monthly_sales, 
            x='Month', 
            y='Sales', 
            markers=True, 
            title="Monthly Sales Trend: Seasonality Impact",
            labels={'Sales': 'Total Revenue ($)', 'Month': 'Month'}
        )
        fig_monthly.update_traces(line_color='green', line_width=3, marker=dict(size=8))
        st.plotly_chart(fig_monthly, width="stretch")
        
        st.info("💡 **Insight:** Notice the sharp increases typically occurring in August (Back to School) and November/December (Holidays)!")
        
        st.markdown("---")
        
        st.subheader("Top Selling Products")
        top_products = filtered_df.groupby('Product Name')['Sales'].sum().reset_index().sort_values(by='Sales', ascending=False).head(10)
        fig_top_prod = px.bar(
            top_products, 
            x='Sales', 
            y='Product Name', 
            orientation='h', 
            title="Top 10 Products by Revenue",
            color='Sales', 
            color_continuous_scale='Blues'
        )
        fig_top_prod.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_top_prod, width="stretch")
        
        st.markdown("---")
        
        st.subheader("Profitability Across Years")
        yearly_summary = filtered_df.groupby('Year').agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Profit Margin (%)': 'mean'
        }).reset_index()
        
        fig_profit = px.line(
            yearly_summary, 
            x='Year', 
            y='Profit Margin (%)', 
            markers=True, 
            title='Average Profit Margin (%) per Year',
            hover_data=['Sales', 'Profit']
        )
        fig_profit.update_xaxes(type='category')
        fig_profit.update_traces(line_color='orange', line_width=4, marker=dict(size=10, symbol="diamond"))
        st.plotly_chart(fig_profit, width="stretch")

elif page == "Forecast":
    st.title("🚀 Sales Forecast")
    st.markdown("Machine learning projection for future sales trends using Linear Regression.")
    
    st.warning("Note: The forecast model uses all historical data (ignoring filters) to preserve time-series integrity.")
    
    df['YearMonth'] = df['Order Date'].dt.to_period('M')
    ts_data = df.groupby('YearMonth')['Sales'].sum().reset_index()
    ts_data['YearMonth_Str'] = ts_data['YearMonth'].astype(str)
    ts_data['MonthIndex'] = np.arange(len(ts_data))
    
    X = ts_data[['MonthIndex']]
    y = ts_data['Sales']
    
    model = LinearRegression()
    model.fit(X, y)
    
    ts_data['Trend'] = model.predict(X)
    
    st.markdown("### Forecast Settings")
    forecast_months = st.slider("Select number of future months to project", min_value=1, max_value=24, value=12)
    
    last_index = ts_data['MonthIndex'].max()
    future_indices = pd.DataFrame({'MonthIndex': np.arange(last_index + 1, last_index + 1 + forecast_months)})
    future_sales = model.predict(future_indices)
    
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

    # Filter out empty or all-NA columns before concatenation to fix FutureWarnings
    valid_ts_cols = ts_data[['YearMonth_Str', 'Sales', 'Trend', 'Type']].dropna(axis=1, how='all')
    valid_forecast_cols = forecast_df.dropna(axis=1, how='all')
    
    # We maintain Sales column as NaN in forecast directly, it's safe to concat columns explicitly
    combined_df = pd.concat([ts_data[['YearMonth_Str', 'Sales', 'Trend', 'Type']], forecast_df], ignore_index=True)
    
    fig_forecast = px.line(
        combined_df, 
        x='YearMonth_Str', 
        y='Sales', 
        markers=True, 
        title="Historical Sales vs Trend Projection",
        labels={'YearMonth_Str': 'Date (Year-Month)', 'Sales': 'Total Sales ($)'}
    )
    fig_forecast.add_scatter(
        x=combined_df['YearMonth_Str'], 
        y=combined_df['Trend'], 
        mode='lines', 
        name='Linear Trend', 
        line=dict(dash='dash', color='red', width=3)
    )
    
    st.plotly_chart(fig_forecast, width="stretch")
    
    st.success("The model successfully analyzed historical monthly sales and projected linear growth over the requested future timeframe.")
