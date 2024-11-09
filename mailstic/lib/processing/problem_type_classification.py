from .utils import text_lemmatizing, get_emb_by_modele
import pickle
import numpy as np
from pathlib import Path
import os
from .few_shot_inference.inference import detect_defect_type
my_path = Path(__file__).parent

clf = pickle.load(open(os.path.join(my_path, "models/problems_type/clf.pickle"), "rb"))
vectorizer = pickle.load(open(os.path.join(my_path, "models/problems_type/tfidf.pickle"), "rb"))
svd = pickle.load(open(os.path.join(my_path, "models/problems_type/svd.pickle"), "rb"))


def classify_problem_type(text, problem_type_model=None):
    if problem_type_model is None:
        wv2_emb = get_emb_by_modele(text).tolist()
        svd_vec = svd.transform(vectorizer.transform([text_lemmatizing(text)]).toarray()).tolist()[0]
        inp = svd_vec + wv2_emb

        probs = clf.predict_proba(inp)
        print(probs)
        if np.max(probs) < 0.2:
            return "Уточнить"

        return clf.predict(inp)[0]
    else:
        return detect_defect_type(text, model_name=problem_type_model)

