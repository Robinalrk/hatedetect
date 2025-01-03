import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from pyswarm import pso
from geneticalgorithm import geneticalgorithm as ga
from skfuzzy import control as ctrl
import skfuzzy as fuzz

import pickle

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize the lemmatizer and stopwords
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load Davidson dataset (adjust the file path if needed)
df = pd.read_csv('labeled_data.csv')

# Drop unnecessary columns
df = df.drop(columns=['Unnamed: 0', 'count', 'hate_speech', 'offensive_language', 'neither'])

# Check class distribution
print("Class distribution:\n", df['class'].value_counts())

# Function to clean and preprocess text
def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only alphabetic characters
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.lower() not in stop_words]
    return ' '.join(tokens)

# Preprocess the tweet column
df['cleaned_tweet'] = df['tweet'].apply(preprocess_text)

# Plot the class distribution
df['class'].value_counts().plot(kind='bar', color=['skyblue', 'orange'])
plt.title('Class Distribution')
plt.xlabel('Label')
plt.ylabel('Frequency')
plt.xticks([0, 1, 2], ['Class 0', 'Class 1', 'Class 2'], rotation=0)
plt.show()

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_tweet']).toarray()

# Prepare the labels
y = df['class'].values

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize SMOTE to handle class imbalance
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# --------------------- Random Forest Model ---------------------

# Train a Random Forest model with the resampled data (baseline model)
rf_clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_clf.fit(X_resampled, y_resampled)

# Make predictions and evaluate the model on the test set
y_pred_rf = rf_clf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

# --------------------- Fuzzy Logic Integration ---------------------

# Fuzzy Logic for Classification Refinement
def apply_fuzzy_logic(predictions, probas):
    # Create fuzzy variables for class confidence levels
    confidence = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'confidence')
    final_decision = ctrl.Consequent(np.arange(0, 2), 'final_decision')

    # Membership functions
    confidence['low'] = fuzz.trimf(confidence.universe, [0, 0, 0.5])
    confidence['medium'] = fuzz.trimf(confidence.universe, [0, 0.5, 1])
    confidence['high'] = fuzz.trimf(confidence.universe, [0.5, 1, 1])

    final_decision['class_0'] = fuzz.trimf(final_decision.universe, [0, 0, 1])
    final_decision['class_1'] = fuzz.trimf(final_decision.universe, [0, 1, 2])

    # Fuzzy rules for decision refinement
    rule1 = ctrl.Rule(confidence['low'], final_decision['class_0'])
    rule2 = ctrl.Rule(confidence['medium'], final_decision['class_1'])
    rule3 = ctrl.Rule(confidence['high'], final_decision['class_1'])

    # Control system
    classification_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
    classification_system = ctrl.ControlSystemSimulation(classification_ctrl)

    # Apply fuzzy logic
    final_predictions = []
    for i in range(len(probas)):
        classification_system.input['confidence'] = probas[i][1]  # The probability of class 1
        classification_system.compute()
        final_predictions.append(int(round(classification_system.output['final_decision'])))

    return np.array(final_predictions)

# Get prediction probabilities (use for fuzzy logic)
y_proba_rf = rf_clf.predict_proba(X_test)

# Apply fuzzy logic to refine the predictions
y_pred_fuzzy = apply_fuzzy_logic(y_pred_rf, y_proba_rf)

# Evaluate the fuzzy-enhanced predictions
print("Fuzzy Logic Enhanced Accuracy:", accuracy_score(y_test, y_pred_fuzzy))
print("Fuzzy Logic Enhanced Classification Report:\n", classification_report(y_test, y_pred_fuzzy))

# --------------------- Hyperparameter Optimization using PSO ---------------------

def fitness_function_pso(params):
    n_estimators = int(params[0])
    max_depth = int(params[1])
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, class_weight='balanced', random_state=42)
    rf.fit(X_resampled, y_resampled)
    y_pred = rf.predict(X_test)
    return -accuracy_score(y_test, y_pred)

lb = [10, 1]  # Lower bounds for n_estimators and max_depth
ub = [200, 20]  # Upper bounds for n_estimators and max_depth

best_params_pso, _ = pso(fitness_function_pso, lb, ub, swarmsize=10, maxiter=5)

# Train Random Forest with optimized parameters
rf_pso = RandomForestClassifier(n_estimators=int(best_params_pso[0]), max_depth=int(best_params_pso[1]), class_weight='balanced', random_state=42)
rf_pso.fit(X_resampled, y_resampled)
y_pred_pso = rf_pso.predict(X_test)

print("PSO Optimized Random Forest Accuracy:", accuracy_score(y_test, y_pred_pso))
print("PSO Optimized Random Forest Classification Report:\n", classification_report(y_test, y_pred_pso))

# --------------------- Hyperparameter Optimization using GA ---------------------

def fitness_function_ga(params):
    n_estimators = int(params[0])
    max_depth = int(params[1])
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, class_weight='balanced', random_state=42)
    rf.fit(X_resampled, y_resampled)
    y_pred = rf.predict(X_test)
    return -accuracy_score(y_test, y_pred)

ga_model = ga(
    function=fitness_function_ga,
    dimension=2,
    variable_type='int',
    variable_boundaries=np.array([[10, 200], [1, 20]])
)

ga_model.run()

best_params_ga = ga_model.output_dict['variable']

# Train Random Forest with GA-optimized parameters
rf_ga = RandomForestClassifier(n_estimators=int(best_params_ga[0]), max_depth=int(best_params_ga[1]), class_weight='balanced', random_state=42)
rf_ga.fit(X_resampled, y_resampled)
y_pred_ga = rf_ga.predict(X_test)
a
print("GA Optimized Random Forest Accuracy:", accuracy_score(y_test, y_pred_ga))
print("GA Optimized Random Forest Classification Report:\n", classification_report(y_test, y_pred_ga))



# Save model, vectorizer, and preprocessing utilities
with open('random_forest_model.pkl', 'wb') as model_file:
    pickle.dump(rf_clf, model_file)

with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

with open('stopwords.pkl', 'wb') as stopwords_file:
    pickle.dump(stop_words, stopwords_file)

with open('lemmatizer.pkl', 'wb') as lemmatizer_file:
    pickle.dump(lemmatizer, lemmatizer_file)

