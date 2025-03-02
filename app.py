from flask import Flask, request, jsonify
import RNN_model
import requests
import torch
import torch.nn as nn
from torch.optim import Adam

app = Flask(__name__)

allowed_files = {'xlsv', 'csv'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_files

@app.route('/prediction', methods=['POST'])
def prediction():
    if request.method == "POST":
        file = request.files.get("file")
        if file is None or file.filename == "":
            return jsonify({"error": "No file uploaded"})
        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed"})
        try:
            # Configure device to CPU or CUDA
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Load data and transform into tensors
            filename = input("Enter the file path: ")
            no_of_days = int(input("Enter the number of days to predict: "))
            X_train, X_test, y_train, y_test, df, data_min, data_max = RNN_model.load_and_preprocess_data(filename, window_size=30)
            train_loader, test_loader = RNN_model.prepare_data_loaders(X_train, X_test, y_train, y_test)

            # Initialize model, loss function, and optimizer
            model = RNN_model.Weight_Model().to(device)
            loss_fn = nn.MSELoss()
            optimizer = Adam(model.parameters(), lr=0.001)

            # Train on data
            RNN_model.train_model(model, train_loader, loss_fn, optimizer, num_epochs=1000)

            # Evaluate model
            RNN_model.evaluate_model(model, test_loader, loss_fn, data_min, data_max)

            # Retrieve historical weights and predict future weight
            historical_weights = df['Weight'].values
            weight_tensor = torch.tensor(historical_weights).to(torch.float32)

            new_data = weight_tensor[-no_of_days:]
            predicted_weight = RNN_model.predict_weight(model, new_data, data_min, data_max)

            return jsonify({'prediction': predicted_weight})
        except:
            return jsonify({"error": 'error during prediction'})

    return jsonify({'prediction': 0})
