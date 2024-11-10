import re
from typing import Optional, List, Union

import numpy as np

from lib.processing.generate_reg_from_pattern import generate_regex

DEFAULT_PATTERN = r"[A-Za-zА-Яа-я]*\d{6,}[A-Za-zА-Яа-я0-9]*"


def to_latin(word: str) -> str:
    """
    Преобразует кириллические символы в латинские для похожих букв.

    Args:
        word (str): Входное слово для преобразования.

    Returns:
        str: Слово, в котором кириллические буквы заменены на похожие латинские.
    """

    dict_ = {
        "А": "A",
        "В": "B",
        "С": "C",
        "Е": "E",
        "Р": "P",
        "К": "K",
        "Х": "X",
        "О": "O",
        "Н": "H",
        "М": "M",
        "Т": "T",
    }
    new_word = ""
    for letter in word:
        if letter.isalpha() and not (ord("A") <= ord(letter) <= ord("z")):
            if letter in dict_:
                new_word += dict_[letter]
            else:
                new_word += letter
        else:
            new_word += letter
    return new_word


def extract_all_serial_numbers(
    text: str, pattern: Optional[str] = None, max_count_serial_numbers: int = 1
) -> List[Union[str, None]]:
    """
    Извлекает все серийные номера из текста в соответствии с заданным паттерном.

    Args:
        text (str): Текст для поиска серийных номеров.
        pattern (Optional[str]): Регулярное выражение для поиска серийных номеров. Если None, используется DEFAULT_PATTERN.
        max_count_serial_numbers (int): Максимальное количество серийных номеров для извлечения.

    Returns:
        List[Union[str, None]]: Список извлечённых серийных номеров или [None], если ничего не найдено.
    """

    if pattern is None:
        pattern = DEFAULT_PATTERN
    else:
        pattern = generate_regex(pattern)
        print("generating pattern: ", pattern)

    serial_numbers = re.findall(pattern, text)

    filtered_serial_numbers = [
        sn for sn in serial_numbers if not sn.isdigit() and len(sn) > 7
    ]

    serial_numbers_array = np.array(filtered_serial_numbers)

    serial_numbers_array = [to_latin(x.upper()) for x in serial_numbers_array]
    if len(serial_numbers_array) == 0:
        return [None]
    return list(set(serial_numbers_array[:max_count_serial_numbers]))


def extract_serial_number(text, pattern=None):
    """
    Извлекает один серийный номер из текста в соответствии с заданным паттерном.

    Args:
        text (str): Текст для поиска серийного номера.
        pattern (Optional[str]): Регулярное выражение для поиска серийного номера. Если None, используется DEFAULT_PATTERN.

    Returns:
        str: Найденный серийный номер или "Уточнить", если ничего не найдено.
    """

    res = extract_all_serial_numbers(text, pattern, 1)[0]
    return res if res is not None else "Уточнить"


def extract_serial_number_by_patterns(text, patterns=None):
    """
    Извлекает серийные номера из текста по ВСЕМ заданным паттернам.

    Args:
        text (str): Текст для поиска серийных номеров.
        patterns (Optional[List[str]]): Список регулярных выражений для поиска серийных номеров. Если None, используется [DEFAULT_PATTERN].

    Returns:
        Set[str]: Множество найденных серийных номеров.
    """

    if patterns is None:
        patterns = [DEFAULT_PATTERN]

    res = []
    for pattern in patterns:
        res.extend(
            [
                i
                for i in extract_all_serial_numbers(
                    text, pattern, max_count_serial_numbers=20
                )
                if i is not None
            ]
        )

    return set(res)
