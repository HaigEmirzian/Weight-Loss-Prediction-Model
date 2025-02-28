from flask import Flask, request, jsonify
import RNN_model
import requests

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
        # Load weights
        # Transform data into tensor
        # Train on data
        # Predict future weight
        # Return as json
            return jsonify({'prediction': 0})
        except:
            return jsonify({"error": 'error during prediction'})

    return jsonify({'prediction': 0})