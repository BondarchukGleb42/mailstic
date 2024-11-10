import os
from pathlib import Path
import sqlite3

db_path = "lib/users.db"


def add_user(login, password):
    """
    Добавляет нового пользователя в базу данных с указанным логином и паролем.

    Args:
        login (str): Логин пользователя.
        password (str): Пароль пользователя.

    Returns:
        None
    """

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO users (login, password)
        VALUES (?, ?)
    """,
        (login, password),
    )

    conn.commit()
    conn.close()


def check_user(login, password):
    """
    Проверяет данные пользователя (логин и пароль) в базе данных.

    Args:
        login (str): Логин пользователя.
        password (str): Пароль пользователя.

    Returns:
        int: Код результата проверки:
            0 — пользователя не существует,
            1 — пароль неверный,
            2 — данные верные.
    """

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT password FROM users WHERE login = ?", (login,))
    result = cursor.fetchone()

    conn.close()

    if result is None:
        return 0
    elif result[0] != password:
        return 1
    else:
        return 2


def login_exists(login):
    """
    Проверяет, существует ли пользователь с указанным логином.

    Args:
        login (str): Логин пользователя.

    Returns:
        bool: True, если логин существует в базе данных, иначе False.
    """

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT 1 FROM users WHERE login = ?", (login,))
    result = cursor.fetchone()

    conn.close()

    return result is not None


def add_user_audio(login, audio_hash, created_at):
    """
    Добавляет информацию о звуковом файле пользователя в базу данных.

    Args:
        login (str): Логин пользователя.
        audio_hash (str): Хеш аудиофайла.
        created_at (str): Время создания записи.

    Returns:
        None
    """

    conn = sqlite3.connect("audios.db")
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO users (login, audio_hash, created_at)
        VALUES (?, ?, ?)
    """,
        (login, audio_hash, created_at),
    )

    conn.commit()
    conn.close()


def get_audio_hashes_by_login(login):
    """
    Извлекает хеши аудиофайлов и время их создания для указанного пользователя.

    Args:
        login (str): Логин пользователя.

    Returns:
        list[tuple[str, str]]: Список кортежей (audio_hash, created_at).
    """

    conn = sqlite3.connect("audios.db")
    cursor = conn.cursor()

    cursor.execute(
        """
            SELECT audio_hash, created_at FROM users WHERE login = ?
        """,
        (login,),
    )
    result = cursor.fetchall()

    conn.close()
    return [(row[0], row[1]) for row in result]


def add_score(audio_hash, r_score, g_score, speed_score, total_score):
    """
    Добавляет информацию о баллах для аудиофайла в базу данных.

    Args:
        audio_hash (str): Хеш аудиофайла.
        r_score (int): Балл за правильность.
        g_score (int): Балл за грамотность.
        speed_score (int): Балл за скорость.
        total_score (int): Общий балл.

    Returns:
        None
    """

    conn = sqlite3.connect("scores.db")
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO scores (audio_hash, r_score, g_score, speed_score, total_score)
        VALUES (?, ?, ?, ?, ?)
    """,
        (audio_hash, r_score, g_score, speed_score, total_score),
    )

    conn.commit()
    conn.close()


def get_score_by_audio_hash(audio_hash):
    """
    Извлекает баллы для аудиофайла по его хешу.

    Args:
        audio_hash (str): Хеш аудиофайла.

    Returns:
        dict[str, int] | None: Словарь с баллами (r_score, g_score, speed_score, total_score),
        если хеш существует в базе данных, иначе None.
    """

    conn = sqlite3.connect("scores.db")
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT audio_hash, r_score, g_score, speed_score, total_score
        FROM scores
        WHERE audio_hash = ?
    """,
        (audio_hash,),
    )

    result = cursor.fetchone()
    conn.close()

    if result:
        return {
            "r_score": result[1],
            "g_score": result[2],
            "speed_score": result[3],
            "total_score": result[4],
        }
    else:
        return None


def clear_table(table_name):
    """
     Очищает таблицу в базе данных.

     Args:
         table_name (str): Имя таблицы, которую необходимо очистить.

     Returns:
         None
     """

    conn = sqlite3.connect(table_name)
    cursor = conn.cursor()

    try:
        cursor.execute(f"DELETE FROM users")
    except:
        cursor.execute(f"DELETE FROM scores")

    conn.commit()
    conn.close()


def delete_audio_by_hash_and_login(audio_hash, login):
    """
    Удаляет запись о звуковом файле по хешу и логину пользователя.

    Args:
        audio_hash (str): Хеш аудиофайла.
        login (str): Логин пользователя.

    Returns:
        None
    """

    conn = sqlite3.connect("audios.db")
    cursor = conn.cursor()

    cursor.execute(
        """
        DELETE FROM users WHERE audio_hash = ? AND login = ?
    """,
        (audio_hash, login),
    )

    conn.commit()
    conn.close()
