# Weight Loss Prediction Model 💪

## Overview

This project leverages time series forecasting techniques to predict future weight changes based on historical weight data from my weight loss journey. The core model uses deep learning with an RNN (Recurrent Neural Network) to predict future weight trends, based on sliding window-based sequences of past weight data.

## Features

- **Time Series Forecasting:** The model predicts future weight changes based on past data by utilizing a sliding window approach.
  
- **RNN Model:** The main predictive model used is a Recurrent Neural Network (RNN), specifically an LSTM (Long Short-Term Memory) network, known for its effectiveness in sequence prediction problems. The model is trained to predict future weight based on past weight data, and uses Mean Squared Error (MSE) as its loss function.

- **Root Mean Squared Error (RMSE):** The model's performance is evaluated using RMSE, and the results are presented in pounds, giving you an understandable measure of prediction accuracy.

- **Data Normalization:** The weight data is normalized to ensure the model can handle varying scales, and the predictions are then denormalized to reflect the original scale.

## Installing Dependencies
Before running the project, make sure you have the following dependencies installed:


  ```bash
 pip install pandas numpy torch scikit-learn
  ```


## Getting Started

1. **Download the dataset:** The dataset should be in an Excel file format, containing historical weight data. The weight data should be in a column labeled "Weight."
2. **Preprocess the data:** The data will be processed to create sequences of historical weight data (sliding window) and prepare it for training.
3. **Train the model:** The RNN model will be trained on the processed data. You can adjust the number of epochs to optimize training, with the model evaluating itself on test data using RMSE.
4. **Predict future weight:** Once the model is trained, it will predict future weight values based on the most recent historical data.

To run the model and receive a prediction 30 days out, execute the following command:
```
python RNN_model.py
```

This updated repo contains the transition from a SARIMA-based approach found in [WL_Model.py](/WL_Model.py) to an RNN model for improved weight prediction accuracy.

**SARIMA MAE:** 2.40 pounds

**RNN RMSE:** 1.02 pounds
