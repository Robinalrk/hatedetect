# Import necessary libraries
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

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
    # Remove URLs and special characters
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only alphabetic characters
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize the words
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.lower() not in stop_words]
    
    return ' '.join(tokens)

# Preprocess the tweet column
df['cleaned_tweet'] = df['tweet'].apply(preprocess_text)

# Display the first few rows of the cleaned tweet
print(df[['tweet', 'cleaned_tweet']].head())

# Plot the class distribution
df['class'].value_counts().plot(kind='bar', color=['skyblue', 'orange'])
plt.title('Class Distribution')
plt.xlabel('Label')
plt.ylabel('Frequency')
plt.xticks([0, 1], ['Non-Hate Speech', 'Hate Speech'], rotation=0)
plt.show()

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_tweet']).toarray()

# Prepare the labels
y = df['class'].values

# Check the shape of the feature matrix
print(f"Feature matrix shape: {X.shape}")

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Logistic Regression classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
clf_weighted = LogisticRegression(max_iter=1000, class_weight='balanced')
clf_weighted.fit(X_train, y_train)

# Make predictions
y_pred_weighted = clf_weighted.predict(X_test)

# Evaluate the model
print("Accuracy (with class weights):", accuracy_score(y_test, y_pred_weighted))
print("Classification Report (with class weights):\n", classification_report(y_test, y_pred_weighted))



# Initialize SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)

# Apply SMOTE to balance the data
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Train a Logistic Regression model with resampled data
clf_resampled = LogisticRegression(max_iter=1000)
clf_resampled.fit(X_resampled, y_resampled)

# Make predictions
y_pred_resampled = clf_resampled.predict(X_test)

# Evaluate the model
print("Accuracy (with SMOTE):", accuracy_score(y_test, y_pred_resampled))
print("Classification Report (with SMOTE):\n", classification_report(y_test, y_pred_resampled))


rf_clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_clf.fit(X_train, y_train)

y_pred_rf = rf_clf.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Step 2: Train a Random Forest model with the resampled data
rf_clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_clf.fit(X_resampled, y_resampled)

# Step 3: Make predictions and evaluate the model on the test set
y_pred_rf = rf_clf.predict(X_test)

# Step 4: Evaluate the performance
print("Random Forest with SMOTE Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest with SMOTE Classification Report:\n", classification_report(y_test, y_pred_rf))

