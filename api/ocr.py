import paddleocr.paddleocr
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import matplotlib.pyplot as plt


class SerialNumberOCR:
    """
    Класс для получения текста с изображения с информацией о устройстве
    """
    def __init__(self):
        self.ocr_core = PaddleOCR(use_angle_cls=True, lang='en')

    def get_text_from_img(self, image_path):
        """
        Получаем весь текст - различные блоки текста разделяем пробелами
        :param image_path: путь до изображения
        :return: весь текст, распознанный на изображении - чтобы в нём найти SN след. алгоритмом
        """

        result = self.ocr_core.ocr(image_path, cls=True)

        full_text_for_next_stage = ""

        for line in result:
            for word_info in line:
                word, confidence = word_info[1][0], word_info[1][1]
                full_text_for_next_stage += word + " "

        return full_text_for_next_stage


ocr = SerialNumberOCR()


def extract_text_from_img(img_path):
    res_text = ocr.get_text_from_img(img_path)
    return res_text
