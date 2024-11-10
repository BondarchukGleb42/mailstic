import pymorphy3
import re
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from pathlib import Path
import os

my_path = Path(__file__).parent

w2v_vec = Word2Vec.load(os.path.join(my_path, "models/wv2_lematized_16.model"))
morph = pymorphy3.MorphAnalyzer()


def text_lemmatizing(text):
    """
    Лемматизирует текст, удаляет ненужные символы и приводит все слова к нормальной форме.

    Функция обрабатывает текст: убирает специальные символы, цифры и приводит слова к их нормальной форме
    с использованием библиотеки pymorphy3.

    Args:
        text (str): Текст, который нужно лемматизировать.

    Returns:
        str: Лемматизированный текст.
    """

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
    """
    Извлекает эмбеддинги из модели Word2Vec для лемматизированного текста.

    Функция принимает текст, лемматизирует его, и затем для каждого лемматизированного слова
    извлекает его векторное представление из обученной модели Word2Vec. Полученные векторы
    усредняются для получения единого вектора, представляющего весь текст.

    Args:
        text (str): Текст, для которого нужно извлечь эмбеддинги.

    Returns:
        np.ndarray: Усреднённый вектор, представляющий текст.
    """

    all_tokens = set(w2v_vec.wv.index_to_key)
    word_vectors_dict = {word: w2v_vec.wv[word] for word in w2v_vec.wv.index_to_key}

    all_emb = [word_vectors_dict[word] for word in text if word in all_tokens]
    emb = np.mean(all_emb, axis=0)

    return emb
