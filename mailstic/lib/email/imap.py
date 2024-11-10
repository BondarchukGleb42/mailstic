import email
import imaplib


class IMAP:
    def __init__(self, user: str, password: str):
        mail = imaplib.IMAP4_SSL("imap.yandex.ru")
        mail.login(user, password)

        self.mail = mail

    def receive(self):
        self.mail.select("inbox")
        status, data = self.mail.uid("search", "UNSEEN")

        for uid in data:
            pass
