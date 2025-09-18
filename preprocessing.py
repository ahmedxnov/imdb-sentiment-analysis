from nltk.tokenize import word_tokenize
from config import *


def clean_text(text: str) -> str:
    for name, pattern, repl in PATTERNS:
        text = pattern.sub(repl, text)
    return text.strip()

def tokenize_text(text: str) -> list[str]:
    return word_tokenize(text)

def normalize_not(tokens: list[str]) -> list[str]:
    for i in range(len(tokens)):
        if tokens[i] == "n't" or tokens[i] == "n’t":
            tokens[i] = "not"
    return tokens

def stopword_removal(tokens: list[str], stopwords: set[str]) -> list[str]:
    filtered_tokens: list[str] = []
    for token in tokens:              
        if token not in stopwords:
            filtered_tokens.append(token)
    return filtered_tokens


def preprocess_text(text: str) -> list[str]:
    stopwords = (BASE - NEGATORS) | ARTIFACTS | AUXILIARIES | PUNCTUATION | ABBREVIATIONS | PLACEHOLDERS | COMMON_WORDS
    return    stopword_removal(normalize_not(tokenize_text(clean_text(text.lower()))),stopwords)
   
  