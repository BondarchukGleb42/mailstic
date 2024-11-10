from dotenv import load_dotenv

load_dotenv()

import os
from pathlib import Path
import sys
import numpy as np
import streamlit as st
import pandas as pd
import streamlit_antd_components as sac
import plotly.express as px
import datetime
import json
import glob

from lib.api import api
from ui.authentification import signin
from lib.processing.message_processing import process_message, generate_answer
from lib.processing.few_shot_inference.new_class_processing import add_new_class
from lib.processing.few_shot_inference.inference import detect_defect_type


if "sidebar_state" not in st.session_state:
    st.session_state.sidebar_state = "expanded"
st.set_page_config(initial_sidebar_state=st.session_state.sidebar_state, layout="wide")


df = pd.read_csv("ui/data/base.csv", index_col=0)


with st.sidebar:
    st.image("ui/assets/logo.png")

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
    other_tables = list(map(lambda x: x.split("/")[-1], glob.glob("ui/data/*")))
    tables = ["Базовый", "Все данные"] + other_tables
    tables.remove("base.csv")
    selected_df = st.selectbox("Выберите набор данных", tables)
    if selected_df == "Базовый":
        df = df.copy()
    elif selected_df == "Все данные":
        df = pd.concat(
            [pd.read_csv(f"ui/data/{table}", index_col=0) for table in other_tables],
            axis=0,
        )
        df.reset_index(inplace=True)
    else:
        df = pd.read_csv(f"ui/data/{selected_df}", index_col=0)
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
            with open(os.path.join(Path(__file__).parent, "tmp.jpeg"), "wb") as f:
                f.write(uploaded_img.getbuffer())
            uploaded_img = os.path.join(Path(__file__).parent, "tmp.jpeg")
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
            st.write("Извлеченные данные:")
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

    model_names = list(
        map(
            lambda x: x.split("/")[-1],
            glob.glob("lib/processing/few_shot_inference/user_models/*"),
        )
    )
    selected_model = col1.selectbox("Выберите модель", ["Стандартная"] + model_names)

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
            for k, (idx, r) in enumerate(tmp_df.iterrows()):
                my_bar.progress(
                    k / len(tmp_df), text="Это займёт какое-то время... или нет"
                )
                result.append(
                    process_message(
                        r["Тема"],
                        r["Описание"],
                        problem_type_model=(
                            selected_model if selected_model != "Стандартная" else None
                        ),
                    )
                )
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
                    tmp_df_res.to_csv(
                        f"ui/data/{str(datetime.datetime.now().isoformat())}.csv"
                    )
                    st.session_state.saved_table = True
                    st.success(
                        "Набор данных успешно сохранён. Вы можете посмотреть его в истории или выбрать в BI дашборде со статистикой"
                    )

elif pages_tree == "База знаний Q&A":
    st.write("### Настройка Q&A базы знаний")
    st.write(
        """В этом разделе можно настроить базу знаний для автоматизации ответов на распространённые вопросы.
    Вы можете добавить к существущим точкам сбоя ответы и рекомендации на распространённые вопросы, для того, чтобы диалоговый ассистент мог предлагать их пользователю после уточнения типа проблемы."""
    )
    st.markdown("------")
    st.markdown("**Точки сбоя**")

    pofs = api.get("/pof").json()

    cols = st.columns((5, 5))

    for idx, pof in enumerate(pofs):
        print(pof)
        with cols[idx % 2]:
            with st.popover(pof["name"], use_container_width=True):
                st.markdown(f"**{pof["name"]}**")

                if st.button(
                    "Удалить тип сбоя из базы знаний", key=f"delete-pof-{pof["slug"]}"
                ):
                    api.delete(f"/pof/{pof["slug"]}")
                    st.rerun()

                qas = api.get(f"/pof/{pof["slug"]}/qa").json()

                st.write("Заготовленные ответы и рекомендации: ")

                for qa in qas:
                    col1, col2 = st.columns((7, 2))
                    col1.write(qa["question"] + qa["answer"])

                    if col2.button("🗑️", key=f"delete-qa-{qa["id"]}"):
                        api.delete(f"/qa/{qa["id"]}")
                        st.rerun()

                    st.markdown("------")

                question = st.text_input(
                    "Введите новый вопрос", key=f"question-qa-{pof["slug"]}"
                )

                answer = st.text_area(
                    "Введите ответ или рекомендацию", key=f"answer-qa-{pof["slug"]}"
                )

                if st.button("Добавить ➕", key=f"create-qa-{pof["slug"]}"):
                    api.post(
                        f"/pof/{pof["slug"]}/qa",
                        json={"question": question, "answer": answer},
                    )
                    st.rerun()

    with st.popover("Добавить тип отказа оборудования ➕", use_container_width=True):
        new_type = st.text_input("Введите название типа отказа")

        if st.button("Применить"):
            api.post("/pof", json={"name": new_type})
            st.rerun()

elif pages_tree == "Настройка типов отказов":
    st.markdown("### Добавление новых типов отказов")
    st.markdown(
        "В этом разделе можно добавить поддержку новую точку отказа на основе нескольких примеров сообщений с такой проблемой."
    )
    col1, col2 = st.columns((2, 3))
    class_name = col1.text_input("Введите название новой точки отказа")
    col2.write("Введите примеры текстов писем с новой точкой отказа **(не менее 5)**")
    if "dataset_text" not in st.session_state:
        st.session_state.dataset_text = []

    with col2:
        page_num = sac.pagination(
            align="left",
            jump=True,
            show_total=True,
            page_size=2,
            total=len(st.session_state.dataset_text),
        )

    for idx, txt in enumerate(
        st.session_state.dataset_text[(page_num - 1) * 2 : page_num * 2]
    ):
        with col2.container(border=True):
            st.write("**Тема**")
            st.write(txt[0])
            st.write("**Сообщение**")
            st.write(txt[1])
            if st.button("Удалить", key=str(idx) + "_dataset"):
                del st.session_state.dataset_text[idx]
                st.rerun()

    with col1.container(border=True):
        input_theme = st.text_input("Введите тему письма")
        input_text = st.text_area("Введите текст письма")
        if st.button("Добавить в датасет ➕"):
            st.session_state.dataset_text.append([input_theme, input_text])
            st.rerun()

    if col1.button("Добавить точку отказа"):
        if len(st.session_state.dataset_text) < 5:
            col1.error("Добавьте минимум 5 примеров с новой точкой отказа")
        elif class_name == "":
            st.error("Пустое название типа отказа")
        else:
            dataset = [x[0] + " " + x[1] for x in st.session_state.dataset_text]
            api.post("/pof/classifier", json={"name": class_name, "dataset": dataset})
            col1.success("Успешно!")

    with col1:
        for _ in range(5):
            st.write("")

        st.write("Вы можете протестировать ваши модели с расширенными классами")
        model_names = list(
            map(
                lambda x: x.split("/")[-1],
                glob.glob("lib/processing/few_shot_inference/user_models/*"),
            )
        )
        selected_model = st.selectbox("Выберите модель", ["Стандартная"] + model_names)

        theme = st.text_input("Тема письма")
        text = st.text_area("Текст письма")
        if st.button("Проверить"):
            text = theme + " " + text
            if selected_model != "Стандартная":
                pred = detect_defect_type(text, selected_model)
            else:
                pred = process_message(theme, text)["problem_type"]
            st.write(f"Предсказанная точка отказа: **{pred}**")
