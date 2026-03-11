# Superstore Analytics Dashboard 🛒📈

A fully interactive, web-based analytics dashboard built with **Streamlit** and **Python** to visualize and forecast sales patterns from a retail store dataset.

## ✨ Features

- **Dashboard**: High-level KPI metrics, sales by category, and regional performance breakdowns using interactive Plotly charts.
- **Analytics**: Deep dive into seasonal trends, monthly sales distributions, top 10 best-selling products, and yearly profitability growth. 
- **Forecast**: Built-in machine learning module featuring a Linear Regression model to project future multi-month sales trends based on historical data.
- **Global Filters**: Dynamic sidebar filters for Year and Region that seamlessly slice the data across the Dashboard and Analytics pages.

## 🚀 Getting Started

Follow these steps to run the application locally.

### 1. Prerequisites 
Ensure you have Python 3.9+ installed on your system.

### 2. Set Up a Virtual Environment (Recommended)
```bash
python -m venv venv

# On Windows:
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
Install all required packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App
Start the development server:
```bash
streamlit run app.py
```
This will open a local web server (usually at `http://localhost:8501`) and automatically launch your default browser.

## 🛠 Built With

- **[Streamlit](https://streamlit.io/)** - For rapid web application development and UI layout.
- **[Pandas](https://pandas.pydata.org/)** - For robust data cleaning, grouping, and feature engineering.
- **[Plotly Express](https://plotly.com/python/plotly-express/)** - For generating interactive and dynamic web charts.
- **[Scikit-Learn](https://scikit-learn.org/)** - For linear regression forecasting within the app.

## 📂 Project Structure
- `app.py`: The main entry point containing all Streamlit frontend logic and model integration.
- `Superstore.ipynb`: Jupyter Notebook detailing initial exploratory data analysis (EDA), data cleaning, and feature engineering.
- `Superstore_Cleaned.xls`: The pre-processed dataset used by the application.
- `requirements.txt`: List of dependencies bound to the application.
