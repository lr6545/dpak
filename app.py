# streamlit_app.py

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import streamlit as st

# Title and Description for the App
st.title('Time Series Forecasting for Building Materials')
st.write("""
This app allows you to analyze and forecast the prices of Cement, Bricks, and TMB over time using ARIMA models.
""")

# Load the dataset
data = pd.read_csv('Datas2018-2025.csv')

# Combine "Year" and "Month" into a datetime column
data['Date'] = pd.to_datetime(data['Year'].astype(str) + ' ' + data['Month'], format='%Y %B')

# Convert "Price (per unit)" to numeric for further processing
data['Price (per unit)'] = pd.to_numeric(data['Price (per unit)'], errors='coerce')

# Create a function to plot and fit ARIMA for each material
def plot_material_time_series(material_name):
    material_data = data[data['Material'] == material_name].copy()
    material_data.set_index('Date', inplace=True)

    # Plot the time series data
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(material_data.index, material_data['Price (per unit)'], label=f'{material_name} Price')
    ax.set_title(f'{material_name} Price Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (per unit)')
    ax.legend()
    ax.grid(True)

    # Show the plot on Streamlit
    st.pyplot(fig)

    # Fit an ARIMA model
    model = ARIMA(material_data['Price (per unit)'], order=(1, 1, 1))
    model_fit = model.fit()

    # Display the ARIMA model summary
    st.subheader(f'ARIMA Model Summary for {material_name}')
    st.text(model_fit.summary())

# Material selection box in Streamlit
material = st.selectbox('Select a Material to Analyze', ('Cement', 'Bricks', 'TMB'))

# Plot and analyze the selected material
plot_material_time_series(material)
