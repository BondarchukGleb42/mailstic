from lib.email.smtp import SMTP
from mailstic.lib.email.message_builder import build_message_from_markdown

smtp = SMTP("mailstic@cfeee1e5e4e00a.ru", "urkhkxxuosptrrip")

markup = """
# Привет!

Я отправил сам себе письмо, оно отформатировано в **Markdown** B-)

### Список

- el1
- el2

Спасибо за внимание!
"""

msg = build_message_from_markdown("Проверка связи", markup)

smtp.send("nerlihmax@yandex.ru", msg)
