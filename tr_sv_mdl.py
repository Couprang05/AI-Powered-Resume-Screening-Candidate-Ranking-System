import pickle
import pandas as pd
from tr_mdl import tr_rsm_rnkr
from feat_ext import ext_feats

data = pd.read_csv("resume_dataset.csv")
resume_texts = data["Resume"]
labels = data["Category"]  

X, vectorizer = ext_feats(resume_texts)

model = tr_rsm_rnkr(X, labels)

with open("resume_ranker.pkl", "wb") as file:
    pickle.dump((model, vectorizer), file)
