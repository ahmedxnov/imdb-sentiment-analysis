#!/usr/bin/env python3

from preprocessing import preprocess_text
import joblib

# Load models
vectorizer = joblib.load("models/tf_idf_vectorizer.joblib")
model = joblib.load("models/LogisticRegression_classifier.joblib")

# Get user input
review = input("Enter a movie review: ")

# Predict sentiment
processed = preprocess_text(review)
features = vectorizer.transform([" ".join(processed)])
prediction = model.predict(features)[0]

print(f"Sentiment: {'Positive' if prediction == 1 else 'Negative'}")