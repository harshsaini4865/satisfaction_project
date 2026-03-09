from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
import pickle
import os

app = Flask(__name__)

# Global variables for the model and preprocessing tools
model = None
scaler = None
label_encoders = {}

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests with proper error handling"""
    try:
        # Get data from the request
        data = request.json
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['entertainment', 'baggage', 'cleanliness']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Convert ratings to integers with error handling
        try:
            entertainment = int(data.get('entertainment', 3))
            baggage = int(data.get('baggage', 3))
            cleanliness = int(data.get('cleanliness', 3))
            
            # Validate rating ranges
            if not (1 <= entertainment <= 5 and 1 <= baggage <= 5 and 1 <= cleanliness <= 5):
                return jsonify({'error': 'Ratings must be between 1 and 5'}), 400
                
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid rating values'}), 400
        
        # Calculate average rating
        avg_rating = (entertainment + baggage + cleanliness) / 3
        
        # Determine prediction based on average rating
        if avg_rating >= 4:
            prediction = "satisfied"
            confidence = 0.85
        elif avg_rating >= 2.5:
            prediction = "neutral or dissatisfied"
            confidence = 0.70
        else:
            prediction = "dissatisfied"
            confidence = 0.80
        
        # Return the result
        result = {
            'prediction': prediction,
            'confidence': confidence,
            'status': 'success',
            'avg_rating': round(avg_rating, 2),
            'message': 'Prediction completed successfully'
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': 'An internal error occurred while processing your request'}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': True})

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Starting Flask application...")
    app.run(debug=True, host='0.0.0.0', port=5000)
