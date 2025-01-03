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
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from pyswarm import pso
from geneticalgorithm import geneticalgorithm as ga
from skfuzzy import control as ctrl
import skfuzzy as fuzz
import pickle

# Initialize NLTK components
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Text preprocessing setup
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load and preprocess dataset
df = pd.read_csv('labeled_data.csv')
df = df.drop(columns=['Unnamed: 0', 'count', 'hate_speech', 'offensive_language', 'neither'])

def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only alphabetic characters
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.lower() not in stop_words]
    return ' '.join(tokens)

df['cleaned_tweet'] = df['tweet'].apply(preprocess_text)

# TF-IDF feature extraction
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_tweet']).toarray()
y = df['class'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# -------------------- Logistic Regression --------------------
log_reg = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
log_reg.fit(X_resampled, y_resampled)
y_pred_lr = log_reg.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_lr))

# -------------------- Fuzzy Logic Integration --------------------
def apply_fuzzy_logic(predictions, probas):
    confidence = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'confidence')
    decision = ctrl.Consequent(np.arange(0, 3), 'decision')

    confidence['low'] = fuzz.trimf(confidence.universe, [0, 0, 0.5])
    confidence['medium'] = fuzz.trimf(confidence.universe, [0.3, 0.5, 0.7])
    confidence['high'] = fuzz.trimf(confidence.universe, [0.5, 1, 1])

    decision['class_0'] = fuzz.trimf(decision.universe, [0, 0, 1])
    decision['class_1'] = fuzz.trimf(decision.universe, [0, 1, 2])
    decision['class_2'] = fuzz.trimf(decision.universe, [1, 2, 2])

    rule1 = ctrl.Rule(confidence['low'], decision['class_0'])
    rule2 = ctrl.Rule(confidence['medium'], decision['class_1'])
    rule3 = ctrl.Rule(confidence['high'], decision['class_2'])

    classification_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
    classification_system = ctrl.ControlSystemSimulation(classification_ctrl)

    refined_predictions = []
    for i in range(len(probas)):
        classification_system.input['confidence'] = max(probas[i])
        classification_system.compute()
        refined_predictions.append(int(round(classification_system.output['decision'])))

    return np.array(refined_predictions)

probas_lr = log_reg.predict_proba(X_test)
y_pred_fuzzy = apply_fuzzy_logic(y_pred_lr, probas_lr)

print("Fuzzy Logic Enhanced Accuracy:", accuracy_score(y_test, y_pred_fuzzy))
print("Fuzzy Logic Enhanced Classification Report:\n", classification_report(y_test, y_pred_fuzzy))

# -------------------- PSO Optimization --------------------
def fitness_function_pso(params):
    lr = LogisticRegression(C=params[0], max_iter=int(params[1]), class_weight='balanced', random_state=42)
    lr.fit(X_resampled, y_resampled)
    y_pred = lr.predict(X_test)
    return -accuracy_score(y_test, y_pred)

lb = [0.01, 100]  # Lower bounds: C, max_iter
ub = [10, 3000]   # Upper bounds: C, max_iter

best_params_pso, _ = pso(fitness_function_pso, lb, ub, swarmsize=5, maxiter=3)
optimized_lr_pso = LogisticRegression(C=best_params_pso[0], max_iter=int(best_params_pso[1]+1000), class_weight='balanced', random_state=42)
optimized_lr_pso.fit(X_resampled, y_resampled)

y_pred_pso = optimized_lr_pso.predict(X_test)
print("PSO Optimized Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_pso))
print("PSO Optimized Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_pso))

# -------------------- GA Optimization --------------------


# Fitness function using a subset of data
def fitness_function_ga(params):
    C = params[0]
    max_iter = int(params[1])
    
    # Use a smaller subset of data for faster computation
    X_sample = X_resampled[:1000]
    y_sample = y_resampled[:1000]
    
    # Train logistic regression
    lr = LogisticRegression(C=C, max_iter=max_iter, class_weight='balanced', random_state=42)
    lr.fit(X_sample, y_sample)
    
    # Evaluate accuracy
    y_pred = lr.predict(X_test)
    return -accuracy_score(y_test, y_pred)  # Negative because GA minimizes the function

# Define GA parameter boundaries
varbound = np.array([[0.01, 10], [100, 500]])

# Initialize and run GA with corrected parameters
ga_model = ga(
    function=fitness_function_ga,
    dimension=2,
    variable_type='real',
    variable_boundaries=varbound,
    algorithm_parameters={
        'max_num_iteration': 5,
        'population_size': 10,
        'parents_portion': 0.3,  # Set portion of parents
        'crossover_probability': 0.8,
        'elit_ratio': 0.1,
        'mutation_probability': 0.2,  
        'crossover_type': 'uniform',
        'max_iteration_without_improv': 2
    }
)

ga_model.run()

# Extract best parameters
best_params_ga = ga_model.output_dict['variable']

# Train final logistic regression model with optimized parameters
optimized_lr_ga = LogisticRegression(C=best_params_ga[0], max_iter=int(best_params_ga[1]), class_weight='balanced', random_state=42)
optimized_lr_ga.fit(X_resampled, y_resampled)

# Evaluate optimized model
y_pred_ga = optimized_lr_ga.predict(X_test)
print("GA Optimized Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_ga))
print("GA Optimized Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_ga))

# -------------------- Save Models --------------------
with open('logistic_regression_model.pkl', 'wb') as model_file:
    pickle.dump(log_reg, model_file)

with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)


# Save y_test and predictions
with open('test_data.pkl', 'wb') as f:
    pickle.dump({'y_test': y_test, 'y_pred_lr': y_pred_lr, 'y_pred_fuzzy': y_pred_fuzzy, 'y_pred_pso': y_pred_pso, 'y_pred_ga': y_pred_ga}, f)
