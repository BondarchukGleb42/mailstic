import re


def generate_regex(pattern):
    ''''
    Генерирует регулярку по правилу, описсание правила:

    после ! знака:
    - E/e - английская строчная/заглавная буква
    - R/r - русская строчная/заглавная буквы
    - D - цифра

    после e/E/r/R может стоять | сразу после - регистр не важен

    например:
        !E!E!E_!D!D!D: <три заглавных английских буквы>_<три цифры>
        !R!R!R_!D!D!D!D: <три заглавных русских буквы>_<четыре цифры>
        !R|_!D <заглвная русская>_<любая цифра>
    '''

    regex = ""
    i = 0
    while i < len(pattern):
        if pattern[i] == '!':
            # слледующий символ после '!' обрабатывается гибко
            i += 1

            if i < len(pattern):
                # проверяем особые символы - буквы и цифру

                if pattern[i] == 'E':
                    if i + 1 < len(pattern) and pattern[i + 1] == '|':
                        regex += "[A-Za-z]"
                        i += 1
                    else:
                        regex += "[A-Z]"

                elif pattern[i] == 'e':
                    if i + 1 < len(pattern) and pattern[i + 1] == '|':
                        regex += "[A-Za-z]"
                        i += 1
                    else:
                        regex += "[a-z]"

                elif pattern[i] == 'R':
                    if i + 1 < len(pattern) and pattern[i + 1] == '|':
                        regex += "[А-Яа-яЁё]"
                        i += 1
                    else:
                        regex += "[А-ЯЁ]"

                elif pattern[i] == 'r':
                    if i + 1 < len(pattern) and pattern[i + 1] == '|':
                        regex += "[А-Яа-яЁё]"
                        i += 1
                    else:
                        regex += "[а-яё]"

                elif pattern[i] == 'D':
                    regex += "[0-9]"
        else:
            regex += re.escape(pattern[i])  # Любые другие символы добавить как есть
        i += 1

    return regex

