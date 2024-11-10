from lib.email.imap import IMAP

imap = IMAP(user="mailstic@cfeee1e5e4e00a.ru", password="urkhkxxuosptrrip")

print(imap.receive())
