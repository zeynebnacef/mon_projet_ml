from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
MODEL_PATH = "gbm_model.joblib"
model = joblib.load(MODEL_PATH)

# Serve the HTML file
@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

# Define a prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        input_data = request.json
        features = input_data['features']

        # Convert input to a numpy array and reshape for prediction
        features_array = np.array(features).reshape(1, -1)

        # Make a prediction
        prediction = model.predict(features_array)
# Log the prediction to MLflow
        with mlflow.start_run():
            mlflow.log_param("input_features", features)
            mlflow.log_metric("prediction", prediction[0])
        # Return the prediction as JSON
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005)
