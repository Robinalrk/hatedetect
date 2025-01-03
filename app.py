from flask import Flask, request, jsonify, render_template
import pickle
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model and vectorizer
try:
    with open('logistic_regression_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
except FileNotFoundError as e:
    print(f"Error loading model or vectorizer: {e}")
    raise

# Load pre-existing test data for evaluation metrics
try:
    with open("test_data.pkl", 'rb') as f:
        tes = pickle.load(f)
        y_test = tes['y_test']
        y_pred_lr = tes['y_pred_lr']
        y_pred_fuzzy = tes['y_pred_fuzzy']
        y_pred_pso = tes['y_pred_pso']
        y_pred_ga = tes['y_pred_ga']
except FileNotFoundError as e:
    print(f"Error loading test data: {e}")
    raise

# Initialize NLTK components
lemmatizer = WordNetLemmatizer()

# Text preprocessing function
def preprocess_text(text):
    if not text or pd.isna(text):  # Handle empty or NaN text
        return []
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only alphabetic characters
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens]
    return tokens

# Highlight tokens based on predictions
def highlight_tokens(tokens):
    highlighted = []
    for token in tokens:
        vectorized_token = vectorizer.transform([token])
        prediction = model.predict(vectorized_token)[0]

        # Apply colors based on prediction
        if prediction == 0:  # Hate Speech
            highlighted.append(f'<b><span style="color: red;">{token}</span></b>')
        elif prediction == 1:  # Offensive Language
            highlighted.append(f'<b><span style="color: green;">{token}</span></b>')
        else:  # Neither
            highlighted.append(token)
    return ' '.join(highlighted)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Compute metrics for display
def get_model_metrics():
    metrics = {
        "Logistic Regression": {
            "Accuracy": accuracy_score(y_test, y_pred_lr),
            "Classification Report": classification_report(y_test, y_pred_lr, output_dict=True)
        },
        "Fuzzy Logic": {
            "Accuracy": accuracy_score(y_test, y_pred_fuzzy),
            "Classification Report": classification_report(y_test, y_pred_fuzzy, output_dict=True)
        },
        "PSO Optimized": {
            "Accuracy": accuracy_score(y_test, y_pred_pso),
            "Classification Report": classification_report(y_test, y_pred_pso, output_dict=True)
        },
        "GA Optimized": {
            "Accuracy": accuracy_score(y_test, y_pred_ga),
            "Classification Report": classification_report(y_test, y_pred_ga, output_dict=True)
        }
    }
    return metrics

# Route to fetch metrics
@app.route('/metrics', methods=['GET'])
def metrics():
    try:
        metrics = get_model_metrics()
        return jsonify(metrics)
    except Exception as e:
        return jsonify({'error': f'Failed to fetch metrics: {str(e)}'}), 500

# Prediction route
@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_text = request.form['tweet']
        if not input_text or pd.isna(input_text):  # Validate input
            return jsonify({'error': 'Invalid input text'}), 400
        
        tokens = preprocess_text(input_text)
        if not tokens:  # Ensure tokens aren't empty
            return jsonify({'error': 'Preprocessing resulted in empty text'}), 400
        
        # Debugging: Print tokens to ensure preprocessing is working
        print("Tokens:", tokens)

        # Highlight tokens dynamically
        highlighted_text = highlight_tokens(tokens)

        # Debugging: Check if vectorized text is working
        vectorized_text = vectorizer.transform([' '.join(tokens)])
        print("Vectorized Text:", vectorized_text.shape)  # Check the shape of the vectorized text
        
        # Predict overall class for the input
        overall_prediction = model.predict(vectorized_text)[0]

        # Debugging: Check the prediction
        print("Prediction:", overall_prediction)

        if overall_prediction is None:
            return jsonify({'error': 'Model prediction failed'}), 500
        
        classes = {0: "Hate Speech", 1: "Offensive Language", 2: "No Hate or Offensive"}
        
        return jsonify({
            'highlighted_input': highlighted_text,
            'prediction': classes.get(overall_prediction, 'Undefined')
        })
    
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
# Route to handle adding new data and retraining model
@app.route('/add-to-dataset', methods=['POST'])
def add_to_dataset():
    try:
        input_text = request.form['userText']
        input_label = request.form.get('userLabel')

        if not input_text or input_label is None:  # Check for missing fields
            return jsonify({'error': 'Missing required fields'}), 400

        input_label = int(input_label)

        # Preprocess input text
        cleaned_text = ' '.join(preprocess_text(input_text))
        if not cleaned_text:  # Handle empty cleaned text
            return jsonify({'error': 'Invalid input text'}), 400

        # Save to CSV
        new_row = {'tweet': input_text, 'cleaned_tweet': cleaned_text, 'class': input_label}
        dataset_path = 'labeled_data.csv'

        if pd.io.common.file_exists(dataset_path):
            df = pd.read_csv(dataset_path)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(dataset_path, mode='w', index=False, header=True)
        else:
            df = pd.DataFrame([new_row])
            df.to_csv(dataset_path, mode='w', index=False, header=True)

        # Trigger retraining
        retrain_model()

        return jsonify({'message': 'Data successfully added to the dataset and model retrained!'})
    
    except ValueError as ve:
        return jsonify({'error': f'Invalid data: {str(ve)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Failed to add data to the dataset: {str(e)}'}), 500

def retrain_model():
    try:
        # Load the labeled dataset for retraining
        dataset = pd.read_csv('labeled_data.csv')
        X = dataset['cleaned_tweet']
        y = dataset['class']

        # Refit the vectorizer and retrain the model
        global vectorizer, model
        vectorizer.fit(X)
        X_vectorized = vectorizer.transform(X)
        model.fit(X_vectorized, y)

        # Save the updated model and vectorizer
        with open('logistic_regression_model.pkl', 'wb') as model_file:
            pickle.dump(model, model_file)
        with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
            pickle.dump(vectorizer, vectorizer_file)

    except Exception as e:
        print(f"Error during model retraining: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
