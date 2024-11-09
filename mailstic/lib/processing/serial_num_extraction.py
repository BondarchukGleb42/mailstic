import re
import numpy as np


def to_latin(word):
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


def extract_serial_number(text):
    pattern = r"[A-Za-zА-Яа-я]*\d{6,}[A-Za-zА-Яа-я0-9]*"

    serial_numbers = re.findall(pattern, text)

    filtered_serial_numbers = [
        sn for sn in serial_numbers if not sn.isdigit() and len(sn) > 7
    ]

    serial_numbers_array = np.array(filtered_serial_numbers)

    serial_numbers_array = [to_latin(x.upper()) for x in serial_numbers_array]
    if len(serial_numbers_array) == 0:
        return None
    return ", ".join(set(serial_numbers_array))
