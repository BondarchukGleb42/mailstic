from smtplib import SMTP_SSL
from email.mime.multipart import MIMEMultipart


class SMTP:
    def __init__(self, user: str, password: str):
        server = SMTP_SSL("smtp.yandex.com")
        server.login(user, password)

        self.server = server
        self.sender = user

    def send(self, recv: str, msg: MIMEMultipart):
        msg["From"] = self.sender
        msg["To"] = recv

        self.server.sendmail(
            self.sender,
            recv,
            msg.as_string(),
        )
