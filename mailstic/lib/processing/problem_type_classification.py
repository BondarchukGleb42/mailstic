from lib.processing.utils import text_lemmatizing, get_emb_by_modele
import pickle
import numpy as np
from pathlib import Path
import os

my_path = Path(__file__).parent

clf = pickle.load(open(os.path.join(my_path, "models/problems_type/clf.pickle"), "rb"))
vectorizer = pickle.load(
    open(os.path.join(my_path, "models/problems_type/tfidf.pickle"), "rb")
)
svd = pickle.load(open(os.path.join(my_path, "models/problems_type/svd.pickle"), "rb"))


def classify_problem_type(text):
    wv2_emb = get_emb_by_modele(text).tolist()
    svd_vec = svd.transform(
        vectorizer.transform([text_lemmatizing(text)]).toarray()
    ).tolist()[0]
    inp = svd_vec + wv2_emb

    probs = clf.predict_proba(inp)
    print(probs)
    if np.max(probs) < 0.2:
        return None

    return clf.predict(inp)[0]
