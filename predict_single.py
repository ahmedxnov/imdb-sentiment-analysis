from preprocessing import preprocess_text
import joblib

# Load models
vectorizer = joblib.load("models/tf_idf_vectorizer.joblib")
model = joblib.load("models/LogisticRegression_classifier.joblib")

# Predict sentiment
review = "This movie was absolutely fantastic!"
processed = preprocess_text(review)
features = vectorizer.transform([" ".join(processed)])
prediction = model.predict(features)[0]

print("Positive" if prediction == 1 else "Negative")