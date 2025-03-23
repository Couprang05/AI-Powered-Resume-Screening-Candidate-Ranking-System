import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def ext_feats(rsm_texts):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    X = vectorizer.fit_transform(rsm_texts)
    return X, vectorizer
