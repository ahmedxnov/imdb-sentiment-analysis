from model import build_model
from data import load_raw_dataset, cpu_info, preprocess_dataset, join_tokens, prepare_labels, create_splits
from vectorizers import build_vectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import yaml
import joblib
import os

if __name__ == "__main__":
    print("Loading configuration...")
    with open("configs/logistic.yaml") as f:
        configs = yaml.safe_load(f)
    print("Configuration loaded.\n")

    print("Loading raw dataset...")
    dataset = load_raw_dataset("dataset/IMDB Dataset.csv")
    print("Dataset loaded.\n")

    print("Getting CPU info...")
    c_count, chunk_size = cpu_info(len(dataset))
    print(f"CPU info retrieved. Cores: {c_count}, Chunk size: {chunk_size}\n")

    print("Preprocessing dataset...")
    X_tokens = preprocess_dataset(dataset["review"].tolist(), c_count, chunk_size)
    print("Dataset preprocessing complete.\n")

    print("Joining tokens...")
    X = join_tokens(X_tokens, dataset.index)
    print("Tokens joined.\n")

    print("Preparing labels...")
    y = prepare_labels(dataset["sentiment"])
    print("Labels prepared.\n")

    print("Creating train/test splits...")
    X_train, X_test, y_train, y_test = create_splits(X, y)
    print("Data splits created.\n")

    print("Building TF-IDF vectorizer...")
    tfidf_params = configs["TF_IDF"]["params"]
    tfidf_params["ngram_range"] = tuple(tfidf_params["ngram_range"])
    tfidf_vectorizer = build_vectorizer(**tfidf_params)
    print("TF-IDF vectorizer built.\n")

    print("Fitting TF-IDF on training data...")
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    print("Transforming test data with TF-IDF...")
    X_test_tfidf  = tfidf_vectorizer.transform(X_test)
    print("TF-IDF fitting and transformation complete.\n")

    print("Building model...")
    model_configs = configs["model"]
    model = build_model(model_configs["name"], **model_configs["params"])
    print("Model built.\n")

    print("Training model...")
    model.fit(X_train_tfidf, y_train)
    print("Model training complete.")
    if(model_configs["name"] != "MultinomialNB"):
        print("Iterations used:", model.n_iter_, '\n')

    print("Evaluating model...\n")
    y_predict = model.predict(X_test_tfidf)
    test_accuracy = accuracy_score(y_predict, y_test)
    
    precision = precision_score(y_test, y_predict) 
    recall = recall_score(y_test, y_predict)
    f1 = f1_score(y_test, y_predict)
    cm = confusion_matrix(y_test, y_predict)
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("Confusion Matrix:\n", cm)
    print(f"Test Accuracy: {test_accuracy*100:.2f}%\n")
    
    
    
    if not os.path.exists("models"):
        os.mkdir("models")
        
    print("Saving fitted TF-IDF vectorizer...")
    joblib.dump(tfidf_vectorizer, "models/tf_idf_vectorizer.joblib")
    print("Vectorizer saved in /models as tf_idf_vectorizer.joblib")

    print("Saving trained classifier model...")
    joblib.dump(model, f"models/{model_configs['name']}_classifier.joblib")
    print(f"Model saved as models/{model_configs['name']}_classifier.joblib")
