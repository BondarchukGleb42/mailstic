import base64
import email
from email.header import decode_header
import imaplib
import os
from uuid import uuid4

from bs4 import BeautifulSoup


class IMAP:
    def __init__(self, user: str, password: str):
        mail = imaplib.IMAP4_SSL("imap.yandex.ru")
        mail.login(user, password)

        self.mail = mail

    def parse_msg(self, msg):
        msg = email.message_from_bytes(msg[0][1])

        sender = msg["Return-path"]
        subject = decode_header(msg["Subject"])[0][0].decode()
        content = []
        img_path = None

        msg.get_payload()

        for part in msg.walk():
            if (
                part.get_content_maintype() == "text"
                and part.get_content_subtype() == "plain"
            ):
                text = base64.b64decode(part.get_payload()).decode(encoding="utf-8")
                text = os.linesep.join(
                    [line.strip() for line in text.splitlines() if line.strip() != ""]
                )
                content.append(text)

            if (
                part.get_content_maintype() == "text"
                and part.get_content_subtype() == "html"
            ):
                soup = BeautifulSoup(
                    part.get_payload(),
                    "html.parser",
                )
                text = soup.get_text()
                text = os.linesep.join(
                    [line.strip() for line in text.splitlines() if line.strip() != ""]
                )
                content.append(text)

            if (
                part.get_content_disposition() == "attachment"
                and part.get_content_maintype() == "image"
            ):
                file_path = f"ui/data/{uuid4()}.img"

                if not os.path.isfile(file_path):
                    with open(file_path, "wb") as f:
                        f.write(part.get_payload(decode=True))

                img_path = file_path

        return (sender, subject, "\n".join(content), img_path)

    def receive(self):
        self.mail.select("inbox")
        _, data = self.mail.uid("search", "UNSEEN")

        mails = []

        for uid in data[0].split():
            _, msg = self.mail.uid("fetch", uid, "(RFC822)")
            mails.append(self.parse_msg(msg))

        return mails

    def close(self):
        self.mail.close()
