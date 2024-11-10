from lib.processing.utils import text_lemmatizing
import pickle
from pathlib import Path
import os

my_path = Path(__file__).parent

DEVICE_TYPE_PATH = "models/device_type/clf.pickle"
TFIDF_PATH = "models/device_type/tfidf.pickle"
SVD_PATH = "models/device_type/svd.pickle"

clf = pickle.load(open(os.path.join(my_path, DEVICE_TYPE_PATH), "rb"))
vectorizer = pickle.load(
    open(os.path.join(my_path, TFIDF_PATH), "rb")
)

svd = pickle.load(open(os.path.join(my_path, SVD_PATH), "rb"))


def classify_device_type(text: str):
    """Классифицирует тип устройства по тексту письма"""
    return clf.predict(
        svd.transform(vectorizer.transform([text_lemmatizing(text)]).toarray())
    )[0][0]
