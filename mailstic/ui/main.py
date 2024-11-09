import os
from pathlib import Path
import numpy as np
import streamlit as st
import pandas as pd
import streamlit_antd_components as sac
import plotly.express as px
import glob
import cv2
import datetime
import time
from ui.authentification import signin
from lib.processing.message_processing import process_message, generate_answer

if "sidebar_state" not in st.session_state:
    st.session_state.sidebar_state = "expanded"
st.set_page_config(initial_sidebar_state=st.session_state.sidebar_state, layout="wide")

df = pd.read_csv(
    os.path.join(Path(__file__).parent, "../lib/processing/data/base.csv"), index_col=0
)

with st.sidebar:
    st.image("assets/logo.png")

    st.markdown(
        """
        <style>
        /* Стили для кнопки */
        div.stButton > button {
            background-color: #F75555;
            color: white;
            border-radius: 8px;
            padding: 0.5em 1em;
        }
        div.stButton > button:hover {
            background-color: #D1DEE8;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns((4, 6))
    if "login" not in st.session_state:
        if col1.button("Войти", use_container_width=True):
            signin()
    else:
        if col1.button("Выйти", use_container_width=True):
            del st.session_state["login"]

    pages_tree = sac.menu(
        items=[
            sac.MenuItem(
                "Главная",
                children=[sac.MenuItem("Статистика"), sac.MenuItem("Загрузка данных")],
                icon=sac.AntIcon("HomeOutlined"),
            ),
            sac.MenuItem(
                "Сообщения",
                children=[
                    sac.MenuItem("Входящие письма"),
                    sac.MenuItem("Тестовое письмо"),
                ],
                icon=sac.AntIcon("TeamOutlined"),
            ),
            sac.MenuItem(
                "Управление алгоритмами",
                children=[
                    sac.MenuItem("Настройка типов отказов"),
                    sac.MenuItem("Настройка серийных номер"),
                ],
            ),
            sac.MenuItem(
                "Инструменты разработчика",
                children=[sac.MenuItem("База знаний Q&A"), sac.MenuItem("API")],
                icon=sac.AntIcon("DatabaseOutlined"),
            ),
        ],
        open_all=True,
        color="red",
    )


if pages_tree == "Главная":
    pages_tree = "Статистика"

if "login" not in st.session_state:
    st.write(":red[Для просмотра этого раздела необходимо авторизоваться]")
elif pages_tree == "Статистика":
    st.markdown("### Статистика сбоев")

    other_tables = list(map(lambda x: x.split("/")[-1], glob.glob("data/*")))
    tables = ["Базовый", "Все данные"] + other_tables
    tables.remove("base.csv")
    selected_df = st.selectbox("Выберите набор данных", tables)
    if selected_df == "Базовый":
        df = df.copy()
    elif selected_df == "Все данные":
        df = pd.concat(
            [pd.read_csv(f"data/{table}", index_col=0) for table in other_tables],
            axis=0,
        )
        df.reset_index(inplace=True)
    else:
        df = pd.read_csv(f"data/{selected_df}", index_col=0)
    df["message_time"] = pd.to_datetime(df["message_time"])
    df.sort_values("message_time", inplace=True)

    start_date, end_date = st.select_slider(
        "Выберите временной интервал",
        options=df["message_time"],
        value=(df["message_time"].iloc[0], df["message_time"].iloc[-1]),
    )
    df = df[(df["message_time"] >= start_date) & (df["message_time"] <= end_date)]

    col1, col2 = st.columns((3, 5))

    fig = px.histogram(
        df["Тип оборудования"],
        title=f"Статистика сбоев в оборудование",
        color_discrete_sequence=["#F75555"],
    )
    fig.update_layout(
        showlegend=False,
        title_font=dict(size=16),
        xaxis_title="Тип оборудования",
        yaxis_title=f"Кол-во обращений",
    )
    col1.plotly_chart(fig, use_container_width=True)

    for _ in range(3):
        col1.write("")

    selected_device = col2.selectbox(
        "Выберите тип оборудования", ["Всё"] + df["Тип оборудования"].unique().tolist()
    )
    if selected_device != "Всё":
        df = df[df["Тип оборудования"] == selected_device]

    fig = px.histogram(
        df["Точка отказа"],
        title=f"Статистика точек отказа",
        color_discrete_sequence=["#F75555"],
    )
    fig.update_layout(
        showlegend=False,
        title_font=dict(size=16),
        xaxis_title="Точка отказа",
        yaxis_title=f"Кол-во обращений",
    )
    col2.plotly_chart(fig, use_container_width=True)


elif pages_tree == "Тестовое письмо":
    st.markdown("### Отправка тестового письма")
    st.write(
        "В этом разделе можно протестировать работу системы в режиме диалога с ассистентом."
    )
    is_restart = st.button("Начать заново")
    if is_restart:
        st.session_state["dialogue_cache"] = []
    if "dialogue_cache" not in st.session_state:
        st.session_state.dialogue_cache = []

    prev_messages = st.empty()

    with st.container(border=True):
        col1, col2 = st.columns((5, 3))
        theme = col1.text_input("Тема письма")
        text = col1.text_area("Текст письма")
        col11, col22 = col1.columns((6, 2))
        uploaded_img = col11.file_uploader(
            "Приложите изображение (опционально) 📎", type=["jpeg"]
        )
        if uploaded_img is not None:
            with open("tmp.jpeg", "wb") as f:
                f.write(uploaded_img.read())
            col22.image("tmp.jpeg", width=100)

        send_button = col1.button("Отправить")

    if send_button:
        if uploaded_img is not None:
            with open("tmp.jpeg", "wb") as f:
                f.write(uploaded_img.getbuffer())
            uploaded_img = "tmp.jpeg"
        else:
            uploaded_img = None

        output = process_message(theme, text, uploaded_img)

        st.session_state.dialogue_cache.append(
            {
                "mail": {"theme": theme, "text": text, "image": "tmp.jpeg"},
                "output": output,
            }
        )
        answer = generate_answer(st.session_state.dialogue_cache)
        with st.chat_message("assistant"):
            st.write(answer["text"])
        if answer["code"] in (0, 1, 4):
            st.write(f"Тип устройства: **{answer['data']['device']}**")
            st.write(f"Точка проблемы: **{answer['data']['problem_type']}**")
            st.write(f"Серийный номер: **{answer['data']['serial_number']}**")

    with prev_messages:
        st.markdown("**Предыдущие сообщения**")
        for message in st.session_state.dialogue_cache:
            with st.container(border=True):
                col1, col2 = st.columns((5, 3))
                col1.write("**Тема письма**")
                col1.write(message["mail"]["theme"])
                col1.write("**Текст письма**")
                col1.write(message["mail"]["text"])

elif pages_tree == "Загрузка данных":
    st.markdown("### Загрузка данных")
    st.markdown(
        "В этом разделе можно загрузить набор писем в csv формате и добавить его в BI-дашборд со статистикой сбоев."
    )

    st.markdown("**Пример csv таблицы:**")
    st.write(
        df.drop(
            ["Тип оборудования", "Точка отказа", "Серийный номер", "index"], axis=1
        ).head()
    )
    st.markdown(
        "**Поле message_time опциональное и содержит в себе дату отправки письма. В случае, если оно не указано в таблице, при добавлении письмам будет присвоена дата загрузки*"
    )

    col1, col2 = st.columns((4, 6))
    uploaded_file = col1.file_uploader(
        "Загрузите csv таблицу с нужными полями", type=["csv"]
    )
    if uploaded_file is not None:
        tmp_df = pd.read_csv(uploaded_file, index_col=0)
        st.session_state.saved_table = False
        if "message_time" not in tmp_df.columns:
            tmp_df["message_time"] = pd.to_datetime(
                pd.Series([str(datetime.datetime.now())] * len(tmp_df))
            )
        col2.write("Загруженный набор данных")
        col2.write(tmp_df)
        if any(
            [
                x in tmp_df.columns
                for x in ["Тип оборудования", "Точка отказа", "Серийный номер"]
            ]
        ):
            col1.warning(
                "В таблице обнаружены столбцы, на месте которых должны находиться предсказания. Они будут удалены."
            )
            for col in ["Тип оборудования", "Точка отказа", "Серийный номер"]:
                if col in tmp_df.columns:
                    tmp_df.drop(col, axis=1, inplace=True)

        if "Тема" not in tmp_df.columns:
            col1.warning("В таблице не обнаружен столбец **Тема**")
        elif "Описание" not in tmp_df.columns:
            col1.warning("В таблице не обнаружен столбец **Описание**")
        else:
            tmp_df["Тема"] = tmp_df["Тема"].fillna("")
            tmp_df["Описание"] = tmp_df["Описание"].fillna("")
            col1.success("Данные успешно загружены. Начинаем генерацию предсказаний")
            my_bar = col1.progress(0, text="Это займёт какое-то время... или нет")
            result = []
            for k, (i, r) in enumerate(tmp_df.iterrows()):
                my_bar.progress(
                    k / len(tmp_df), text="Это займёт какое-то время... или нет"
                )
                result.append(process_message(r["Тема"], r["Описание"]))
            result = pd.DataFrame(result)
            result.rename(
                {
                    "device": "Тип оборудования",
                    "problem_type": "Точка отказа",
                    "serial_number": "Серийный номер",
                },
                axis=1,
                inplace=True,
            )
            tmp_df.reset_index(inplace=True)
            tmp_df_res = pd.concat([tmp_df, result], axis=1)
            tmp_df_res["index"] = np.arange(len(tmp_df_res))
            tmp_df_res = tmp_df_res[
                [
                    "index",
                    "Тема",
                    "Описание",
                    "Тип оборудования",
                    "Точка отказа",
                    "Серийный номер",
                    "message_time",
                ]
            ]

            st.write("Набор данных со сгенерированными предсказаниями:")
            st.write(tmp_df_res)

            if not st.session_state.saved_table:
                if st.button("Сохранить набор данных"):
                    tmp_df_res.to_csv(f"data/{str(datetime.datetime.now())}.csv")
                    st.session_state.saved_table = True
                    st.success(
                        "Набор данных успешно сохранён. Вы можете посмотреть его в истории или выбрать в BI дашборде со статистикой"
                    )
