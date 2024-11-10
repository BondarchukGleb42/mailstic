import email
import imaplib


class IMAP:
    def __init__(self, user: str, password: str):
        mail = imaplib.IMAP4_SSL("imap.yandex.ru")
        mail.login(user, password)
