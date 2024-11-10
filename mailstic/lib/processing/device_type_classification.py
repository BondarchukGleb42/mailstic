from lib.processing.utils import text_lemmatizing
import pickle
from pathlib import Path
import os

my_path = Path(__file__).parent

clf = pickle.load(open(os.path.join(my_path, "models/device_type/clf.pickle"), "rb"))
vectorizer = pickle.load(
    open(os.path.join(my_path, "models/device_type/tfidf.pickle"), "rb")
)
svd = pickle.load(open(os.path.join(my_path, "models/device_type/svd.pickle"), "rb"))


def classify_device_type(text):
    return clf.predict(
        svd.transform(vectorizer.transform([text_lemmatizing(text)]).toarray())
    )[0][0]
