from model import build_model
from data import load_raw_dataset, cpu_info, preprocess_dataset, join_tokens, prepare_labels, create_splits
from vectorizers import build_vectorizer
import yaml

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
    test_accuracy = model.score(X_test_tfidf, y_test)
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
