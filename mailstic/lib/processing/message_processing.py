from .device_type_classification import classify_device_type
from .problem_type_classification import classify_problem_type
from .serial_num_extraction import extract_serial_number
from .ocr import extract_text_from_img
from .qa_answering import get_recommendation


def process_message(theme, text, img_path=None, problem_type_model=None):
    full_text = theme + " " + text
    if img_path is not None:
        ocr_text = extract_text_from_img(img_path)
        num = extract_serial_number(ocr_text)
    else:
        num = extract_serial_number(full_text)

    device = classify_device_type(full_text)
    problem_type = classify_problem_type(full_text, problem_type_model)

    return {
        "device": device,
        "problem_type": problem_type,
        "serial_number": num
    }


def get_serial_code_number_message(device):
    return {
        "Ноутбук": """Для дальнейшего решения проблемы, пожалуйста, отправьте серийный номер ноутбука. 
                    Его можно посмотреть на обратной стороне или на упаковке. Вы также можете отправить фото для автоматического распознавания.""",
        "Сервер": "Для дальнейшего решения проблемы, пожалуйста, отправьте серийный номер сервера. Его можно посмотреть в настройках.",
        "СХД": "Для дальнейшего решения проблемы, пожалуйста, отправьте серийный номер системы хранения данных. Его можно посмотреть в панели управления."
    }[
        device] + " В случае, если мы ошиблись с вашим типом оборудования, пожалуйста, отправьте тип вашего оборудования ответным сообщением."


def get_problem_type_message(device):
    device_to_problem_types = {
        "Ноутбук": ["Аккумулятором", "Камерой", "Клавитурой", "Материнской платой", "Дисплеем"],
        "СХД": ["Диском", "Необходимостью в консультации", "Оперативной памятью"],
        "Сервер": ["SFP модулем", "Программным обеспечением", "Материнской платой"]
    }
    return f"""Пожалуйста, опишите свою проблему подробнее в ответном сообщении.
     В вашем случае это могут быть проблемы с {', '.join(device_to_problem_types[device])} и т.д. В случае, если проблему не удастся выяснить, мы свяжем вас с оператором."""


def generate_answer(dialogue):
    """
    codes:
        0: не удалось узнать серийный номер за 3 сообщения, отправляем оператору;
        1: узнали серийный номер, не удалось узнать проблему за 3 сообщения;
        2: не удалось узнать серийный номер, продолжаем спрашивать;
        3: не удалось узнать тип проблемы, продолжаем спрашивать;
        4: удалось узнать все данные, отправляем оператору
    """
    serial_num_is_none = all([x["output"]["serial_number"] == "Уточнить" for x in dialogue])
    problem_type_is_none = all(x["output"]["problem_type"] == "Уточнить" for x in dialogue)
    device = dialogue[-1]["output"]["device"]
    if len(dialogue) > 3:
        return {
            "text": f"Не удалось распознать {'Сериный номер' if serial_num_is_none else 'Тип проблемы'}. Переключаем вас на оператора тех. поддержки для уточнения обстоятельств.",
            "completed": True,
            "code": 0 if serial_num_is_none else 1,
            "data": {"device": device, "problem_type": "Уточнить", "serial_number": "Уточнить"}
        }
    else:
        if problem_type_is_none:
            if not serial_num_is_none:
                serial_number = [x["output"]["serial_number"] for x in dialogue if x["output"]["serial_number"] != "Уточнить"][-1]
            else:
                serial_number = "Уточнить"
            return {
                "text": get_problem_type_message(device),
                "completed": False,
                "code": 3,
                "data": {"device": device, "problem_type": "Уточнить", "serial_number": serial_number}
            }
        elif serial_num_is_none:
            problem_type = [x["output"]["problem_type"] for x in dialogue if x["output"]["problem_type"] != "Уточнить"][-1]
            return {
                "text": get_serial_code_number_message(device),
                "completed": False,
                "code": 2,
                "data": {"device": device, "problem_type": problem_type, "serial_number": "Уточнить"}
            }
        else:
            text = " ".join([dialogue[0]["mail"]["theme"]] + [x["mail"]["text"] for x in dialogue])
            problem_type = [x["output"]["problem_type"] for x in dialogue if x["output"]["problem_type"] != "Уточнить"][-1]
            serial_number = [x["output"]["serial_number"] for x in dialogue if x["output"]["serial_number"] != "Уточнить"][-1]
            recommendation = get_recommendation(text, problem_type)
            answer = "Нам удалось извлечь все необходимые данные. С вами свяжется специалист технической поддержки в ближайшее время."
            if recommendation is not None:
                answer += "\nВозможно, у вас могут присутствовать следующие проблемы: " + recommendation
            return {
                "text": answer,
                "completed": True,
                "code": 4,
                "data": {"device": device, "problem_type": problem_type, "serial_number": serial_number}
            }

