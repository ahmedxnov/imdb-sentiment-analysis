#!/usr/bin/env python3

from preprocessing import preprocess_text
import joblib
import streamlit as st


@st.cache_resource
def load_models():
    try:
        tfidf_vectorizer = joblib.load("models/tf_idf_vectorizer.joblib")
        model = joblib.load("models/LogisticRegression_classifier.joblib")
        return tfidf_vectorizer, model
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.error("Please train the model first by running train.py")
        return None, None


# Load models
tfidf_vectorizer, model = load_models()
st.info("📂 For more info about this project, visit: https://github.com/ahmedxnov/imdb-sentiment-analysis\n\n")


st.title("Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (Positive or Negative)")





if tfidf_vectorizer is not None and model is not None:
    
    review = st.text_area("Enter a movie review:", height=100, placeholder="Type your movie review here...")
    
    if st.button("Predict Sentiment"):
        if review.strip():
                preprocessed_review = preprocess_text(review)
                test_tf_idf = tfidf_vectorizer.transform([" ".join(preprocessed_review)])
                
                prediction = model.predict(test_tf_idf)
                probability = model.predict_proba(test_tf_idf)[0]

                if prediction[0] == 1:
                    st.success(f" **Positive Sentiment** (Confidence: {probability[1]:.2%})")
                else:
                    st.error(f" **Negative Sentiment** (Confidence: {probability[0]:.2%})")
                st.warning(
    "NOTE: This model struggles with sarcasm, nuance, neutral reviews, or misspellings.\n\n"
    "It also struggles with offensive or profane language.\n\n" 
    "Please keep your reviews clear and direct for the best results.")
        else:
            st.warning("Please enter a movie review before predicting.")