from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

def build_model(name: str, **params):
    if name == "LogisticRegression":
        return LogisticRegression(**params)
    elif name == "SVC":
        return SVC(**params)
    elif name == "MultinomialNB":
        return MultinomialNB(**params)
    elif name == "RandomForestClassifier":
        return RandomForestClassifier(**params)
    else:
        raise ValueError(f"Model '{name}' is not supported.")
    
    
