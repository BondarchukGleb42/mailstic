import json
import os
from pathlib import Path
from fuzzywuzzy import fuzz
import numpy as np
from lib.processing.utils import text_lemmatizing


def get_recommendation(qas, text):
    text = text_lemmatizing(text)

    ratio = [
        fuzz.partial_ratio(
            text, text_lemmatizing(answer["question"] + " " + answer["answer"])
        )
        for answer in qas
    ]

    print("ratios:", ratio)

    if max(ratio) >= 45:
        return qas[np.argmax(ratio)]["answer"]

    return None
