import json
import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from datetime import datetime
import os

app = Flask(__name__)

# Debug: Check if file exists and is accessible
print("Current working directory:", os.getcwd())
print("Files in directory:", os.listdir('.'))
if os.path.exists('complete_blended_model.pkl'):
    print("complete_blended_model.pkl found! File size:", os.path.getsize('complete_blended_model.pkl'), "bytes")
else:
    print("ERROR: complete_blended_model.pkl not found!")

# Load the complete blended model
try:
    with open('complete_blended_model.pkl', 'rb') as f:
        predictor = pickle.load(f)
    print(f"Successfully loaded complete blended model. Type: {type(predictor)}")
    
except Exception as e:
    print(f"Error loading model: {e}")
    print("Creating mock predictor for testing...")
    
    # Create a mock predictor for testing
    class MockPredictor:
        def predict(self, X):
            return np.array([0])  # Always predict Good Loan for testing
            
        def predict_proba(self, X):
            return np.array([[0.7, 0.3]])  # 70% Good, 30% Bad
    
    predictor = MockPredictor()

# Preprocessing functions
def preprocess_term(term):
    if isinstance(term, str):
        return int(term.split()[0])
    return term

def preprocess_earliest_cr_line(date_str):
    try:
        if isinstance(date_str, str):
            date_obj = datetime.strptime(date_str, '%b-%y')
            return date_obj.year
        return date_str
    except:
        return 2000

def preprocess_sub_grade(sub_grade):
    sub_grades = ["A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3", "B4", "B5",
                 "C1", "C2", "C3", "C4", "C5", "D1", "D2", "D3", "D4", "D5",
                 "E1", "E2", "E3", "E4", "E5", "F1", "F2", "F3", "F4", "F5",
                 "G1", "G2", "G3", "G4", "G5"]
    sub_grade_labels = {grade: i for i, grade in enumerate(reversed(sub_grades), start=1)}
    return sub_grade_labels.get(sub_grade, 18)

def preprocess_emp_length(emp_length):
    emp_length_map = {
        '0': 0, '< 1 year': 0, '1 year': 1, '2 years': 2,
        '3 years': 3, '4 years': 4, '5 years': 5, '6 years': 6,
        '7 years': 7, '8 years': 8, '9 years': 9, '10+ years': 10
    }
    return emp_length_map.get(emp_length, 5)

def preprocess_verification_status(status):
    status_map = {'Verified': 0, 'Not Verified': 1, 'Source Verified': 2}
    return status_map.get(status, 1)

def preprocess_input(data_dict):
    processed = data_dict.copy()
    
    # Apply preprocessing
    if 'term' in processed:
        processed['term'] = preprocess_term(processed['term'])
    if 'earliest_cr_line' in processed:
        processed['earliest_cr_line'] = preprocess_earliest_cr_line(processed['earliest_cr_line'])
    if 'sub_grade' in processed:
        processed['sub_grade'] = preprocess_sub_grade(processed['sub_grade'])
    if 'emp_length' in processed:
        processed['emp_length'] = preprocess_emp_length(processed['emp_length'])
    if 'verification_status' in processed:
        processed['verification_status'] = preprocess_verification_status(processed['verification_status'])
    
    # One-hot encoding
    home_ownership = processed.get('home_ownership', 'MORTGAGE')
    purpose = processed.get('purpose', 'debt_consolidation')
    
    for home_type in ['MORTGAGE', 'RENT', 'OWN', 'OTHER']:
        processed[f'home_ownership_{home_type}'] = 1 if home_ownership == home_type else 0
    
    for purpose_type in ['debt_consolidation', 'credit_card', 'home_improvement', 'other']:
        processed[f'purpose_{purpose_type}'] = 1 if purpose == purpose_type else 0
    
    # Remove original columns if they exist
    if 'home_ownership' in processed:
        processed.pop('home_ownership')
    if 'purpose' in processed:
        processed.pop('purpose')
    
    return processed

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/test')
def test():
    return jsonify({
        'message': 'API is working!',
        'model_loaded': predictor is not None,
        'model_type': str(type(predictor)) if predictor else 'None'
    })

@app.route('/loan_api', methods=['POST'])
def loan_api():
    if predictor is None:
        return jsonify({'error': 'Model not loaded', 'success': False}), 500
        
    try:
        request_data = request.get_json()
        if not request_data or 'data' not in request_data:
            return jsonify({'error': 'Missing "data" field in request'}), 400
            
        data = request_data['data']
        print("Received data:", data)
        
        # Preprocess the data
        processed_data = preprocess_input(data)
        print("Processed data keys:", list(processed_data.keys()))
        
        # Convert to DataFrame
        input_df = pd.DataFrame([processed_data])
        print("DataFrame shape:", input_df.shape)
        print("DataFrame columns:", input_df.columns.tolist())
        
        # Make prediction
        prediction = predictor.predict(input_df)
        prediction_proba = predictor.predict_proba(input_df)
        
        result = {
            'prediction': int(prediction[0]),
            'probability_class_0': float(prediction_proba[0][0]),
            'probability_class_1': float(prediction_proba[0][1]),
            'prediction_label': 'Bad Loan' if prediction[0] == 1 else 'Good Loan',
            'success': True
        }
        
        print("Prediction result:", result)
        return jsonify(result)
            
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Error: {str(e)}")
        print(f"Traceback: {error_traceback}")
        return jsonify({'error': str(e), 'success': False}), 400

@app.route('/predict', methods=['POST'])
def predict():
    """Handle form submissions from the HTML frontend"""
    if predictor is None:
        return render_template('home.html', 
                             prediction_text="Error: Model not loaded. Please try again later.")
    
    try:
        # Get form data
        form_data = request.form.to_dict()
        print("Form data received:", form_data)
        
        # Preprocess the form data
        processed_data = preprocess_input(form_data)
        print("Processed data:", processed_data)
        
        # Convert to DataFrame
        input_df = pd.DataFrame([processed_data])
        print("DataFrame shape:", input_df.shape)
        
        # Make prediction
        prediction = predictor.predict(input_df)
        prediction_proba = predictor.predict_proba(input_df)
        
        # Format results for display
        result_class = int(prediction[0])
        result_prob_good = float(prediction_proba[0][0])
        result_prob_bad = float(prediction_proba[0][1])
        result_label = 'BAD LOAN' if result_class == 1 else 'GOOD LOAN'
        
        # Create explanation text
        if result_class == 1:
            explanation = f"This application has a {result_prob_bad*100:.2f}% probability of being a bad loan."
            recommendation = "Recommendation: REJECT application"
            risk_level = "HIGH RISK"
        else:
            explanation = f"This application has a {result_prob_good*100:.2f}% probability of being a good loan."
            recommendation = "Recommendation: APPROVE application"
            risk_level = "LOW RISK"
        
        return render_template('home.html', 
                             prediction_text=f"Loan Prediction: {result_label}",
                             explanation=explanation,
                             recommendation=recommendation,
                             risk_level=risk_level,
                             probability=f"{result_prob_bad*100:.2f}%" if result_class == 1 else f"{result_prob_good*100:.2f}%")
    
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Error processing form: {str(e)}")
        print(f"Traceback: {error_traceback}")
        return render_template('home.html', 
                             prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True, port=8000)