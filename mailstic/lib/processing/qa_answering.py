import json
import os
from pathlib import Path
from fuzzywuzzy import fuzz
import numpy as np
from lib.processing.utils import text_lemmatizing

qa_base = json.load(
    open(os.path.join(Path(__file__).parent, "data/qa_base.json"), "rb")
)


def get_recommendation(text, problem_type):
    if problem_type not in list(qa_base.keys()):
        return None
    answers = qa_base[problem_type]
    text = text_lemmatizing(text)
    ratio = [fuzz.partial_ratio(text, text_lemmatizing(answer)) for answer in answers]
    print("ratios:", ratio)
    if max(ratio) > 45:
        return answers[np.argmax(ratio)]
    return None
