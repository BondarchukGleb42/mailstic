import json
import os
from pathlib import Path
from fuzzywuzzy import fuzz
import numpy as np
from lib.processing.utils import text_lemmatizing


def get_recommendation(qas, text):
    """
    Получает рекомендацию на основе сходства текста запроса с вопросами и ответами в базе данных.

    Функция сравнивает введённый текст с вопросами и ответами из набора вопросов-ответов, используя метод
    частичного сравнения строк (partial ratio) из библиотеки fuzzywuzzy. Если схожесть превышает порог,
    возвращается подходящий ответ.

    Args:
        qas (list[dict]): Список словарей, каждый из которых содержит ключи "question" и "answer" (строки).
        text (str): Текст запроса, для которого нужно найти подходящий ответ.

    Returns:
        Optional[str]: Возвращает строку с ответом, если найдена высокая степень сходства, иначе None.
    """

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
