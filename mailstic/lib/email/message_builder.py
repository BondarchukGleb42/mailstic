from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from markdown import markdown


def build_message_from_markdown(subject: str, msg: str):
    """
    subject - заголовок сообщения
    msg - содержимое письма в Markdown
    """

    message = MIMEMultipart()
    message["Subject"] = subject

    html = markdown(msg)

    message.attach(MIMEText(html, "html"))

    return message
