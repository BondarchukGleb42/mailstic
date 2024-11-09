import pymorphy2
import re
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from pathlib import Path
import os

my_path = Path(__file__).parent

w2v_vec = Word2Vec.load(os.path.join(my_path, "models/wv2_lematized_16.model"))
morph = pymorphy2.MorphAnalyzer()


def text_lemmatizing(text):
    MAX_WORDS_COUNT = 10000000
    if text is None:
        return " "
    text = str(text)
    if len(text) > MAX_WORDS_COUNT:
        return text
    if text is None:
        return None
    text = text.replace("\n", " ")
    text = re.sub("[0-9:,\.!?()-/+*;•$&%]", "", text.lower())
    text = " ".join(
        [morph.parse(word)[0].normal_form for word in text.split()[:MAX_WORDS_COUNT]]
    )
    return text


def get_emb_by_modele(text):
    all_tokens = set(w2v_vec.wv.index_to_key)
    word_vectors_dict = {word: w2v_vec.wv[word] for word in w2v_vec.wv.index_to_key}

    all_emb = [word_vectors_dict[word] for word in text if word in all_tokens]
    emb = np.mean(all_emb, axis=0)

    return emb
