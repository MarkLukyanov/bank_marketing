import streamlit as st
from eda import *
import time
from PIL import Image


def show_main_page():
    # Progress bar
    bar_p = st.progress(0)

    for percentage_complete in range(100):
        time.sleep(0.01)
        bar_p.progress(percentage_complete + 1)


    # Status message
    # display a temporary message when executing a block of code
    with st.spinner('Пожалуйста, подождите...'):
        time.sleep(1)

    bar_p.empty()

    st.title("Отклик на маркетинговое предложение")

    image = Image.open('data/img.png')
    st.image(image)

    st.header("Исследуем наши 'чистые' данные!")

    df = open_data()

    st.subheader("Посмотрим на распределение числовых признаков")
    st.pyplot(explore_numerical(df))
    st.markdown("- Около половины всех взятых кредитов не закрыты.")
    st.markdown("- Более, чем у 10,000 людей нет в собственности машины. Примерно у 10,000 нет и квартиры.")
    st.markdown("- Соотношение мужчин и женщин примерно 2:1.")
    st.markdown("- Основные клиенты банка - граждане от 22 до 60 лет.")
    st.markdown("- На маркетинговое предложение откликнулись 15% всех клиентов.")
    st.markdown("- Кредиты больше 30,000 почти никто не брал.")
    st.markdown("- Почти никто не зарабатывает больше 30,000.")

    st.divider()

    st.subheader("Изучим корреляции")
    df_corr = df.drop(columns=["FAMILY_INCOME", "EDUCATION", "MARITAL_STATUS"])
    st.pyplot(show_correlation(df_corr))
    st.markdown("Наблюдаем очень сильную взаимосвязь между количеством взятых кредитов и количеством закрытых кредитов.")
    st.markdown(" Также высокая корреляция между суммой последнего кредита с первоначальным взносом и его сроком.")

    st.divider()

    st.subheader("Посмотрим на взаимосвязь некоторых столбцов с целевой переменной")
    st.pyplot(show_dependance_on_target(df, "PERSONAL_INCOME"))
    st.text("Видим, что на маркетинговое предложение не откликались люди с доходом больше 150,000.")
    st.pyplot(show_dependance_on_target(df, "AGE"))
    st.markdown("Видим, что самые старые клиенты банка(возрастом более 65 лет) не были заинтересованы в предложениях банка.")

    st.divider()

    st.subheader("Изучим числовые характеристики числовых признаков")
    st.dataframe(show_numerical_features(df))
    st.markdown("Cредний заработок примерно равен среднему значению взятого кредита;"
            "был человек(несколько людей) с 11 кредитами;"
            "огромная разница между минимальным и максимальным заработком(минимальный - 24, максимальный - 250,000)")

    st.divider()

    st.subheader("Изучим числовые характеристики категориальных признаков")
    st.write(explore_categorical(df))


if __name__ == "__main__":
    show_main_page()
