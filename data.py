import pandas as pd
from multiprocessing import Pool
import math
from os import cpu_count
from preprocessing import preprocess_text
from sklearn.model_selection import train_test_split


def load_raw_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def cpu_info(reviews_count: int) -> tuple[int, int]:
    count = cpu_count() or 1
    chunk_size = max(1, math.ceil(reviews_count / (count * 2)))
    return count, chunk_size
    
def preprocess_dataset(reviews: list[str], c_count: int, chunk_size: int) -> list[list[str]]:
    with Pool(c_count) as p:
        processed_texts = p.map(preprocess_text, reviews, chunksize=chunk_size)
    return processed_texts

def join_tokens(processed_texts: list[list[str]], index) -> pd.Series:
    return pd.Series([" ".join(tokens) for tokens in processed_texts], index=index)

def prepare_labels(sentiments: pd.Series) -> pd.Series:
    return sentiments.apply(lambda x: 1 if x == "positive" else 0)

def create_splits(X: pd.Series, y: pd.Series, test_size=0.2, random_state=42) -> tuple:
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test







