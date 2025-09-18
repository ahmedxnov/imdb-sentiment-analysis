from sklearn.linear_model import LogisticRegression

def build_model(name: str, **params):
    if name == "LogisticRegression":
        return LogisticRegression(**params)
    else:
        raise ValueError(f"Model '{name}' is not supported.")
    
    
