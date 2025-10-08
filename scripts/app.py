import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.utils.preprocessing import preprocess_text
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

st.set_page_config(page_title="IMDb Sentiment Analysis", page_icon="🎬", layout="wide")

st.title("🎬 Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (Positive or Negative)")

with st.sidebar:
    st.markdown("## 📋 About")
    st.markdown("This app uses **Logistic Regression** with **TF-IDF** vectorization to classify movie reviews as positive or negative.")
    
    st.markdown("## 📊 Performance")
    st.markdown("- **Accuracy:** 91.62%\n- **F1-Score:** 91.66%\n- **Dataset:** 50,000 IMDb Reviews ")

    
    st.markdown("## 🔧 Key Features")
    st.markdown("- Advanced text preprocessing\n- Negation preservation\n- Custom stopword filtering\n- YAML-based configuration")

    
    st.markdown("## 🔗 Links")
    st.markdown("[📂 GitHub Repository](https://github.com/ahmedxnov/imdb-sentiment-analysis)")
    st.markdown("[📊 Kaggle Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)")
    
    st.markdown("---")
    st.markdown("**Developer:** [Ahmad Khaled](https://www.linkedin.com/in/ahmad-khaled-hamed/)")





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