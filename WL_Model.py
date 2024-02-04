import warnings
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.express as px

# Suppress specific warnings
warnings.filterwarnings("ignore", message="No frequency information was provided", category=UserWarning)

# Finding the right path in life
file_path = r'Weight_Loss_Journey.xlsx'

# Read the Excel file, duhhh
df = pd.read_excel(file_path, engine='openpyxl')

# Data Preparation
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')

# Model Training with SARIMA
order = (1, 1, 1)  # Non-seasonal order (p, d, q)
seasonal_order = (1, 1, 1, 12)  # Seasonal order (P, D, Q, S)

# Create Model
model = SARIMAX(df['Weight'], order=order, seasonal_order=seasonal_order)
trained_model = model.fit(disp=False)

# Get user input for the number of months into the future
num_months = int(input("Enter the number of months into the future you want to see: "))

# Print the Mean Absolute Error (MAE)
mae = trained_model.mae
print(f'Mean Absolute Error (MAE): {mae:.2f} pounds')

# Future predictions for the specified number of months
future_dates = pd.date_range(df.index[-1], periods=num_months * 30, freq='D')  # Assuming 30 days per month
future_predictions = trained_model.get_forecast(steps=num_months * 30).predicted_mean

# Plot/Chart characteristics
fig = px.line(df, x=df.index, y='Weight', labels={'Weight': 'Weight'}, 
              title=f'Weight Prediction Over Time (SARIMA) - Next {num_months} Months',
              line_shape='linear', template='plotly_dark')

# Add a dashed line for future predictions
fig.add_scatter(x=future_dates, y=future_predictions, mode='lines', line=dict(dash='dash'), name='Future Predictions')

# Hover over a date in the chart to show weight at that time
fig.update_traces(hovertemplate='%{y:.2f} lbs', hoverinfo='y')

# Adjust y-axis ticks
fig.update_yaxes(tickmode='auto', nticks=15)

# Reveal the grand prize
fig.show()