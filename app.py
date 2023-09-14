import pickle
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

    tab_1, tab_2 = st.tabs(["Исследуем", "Предсказываем"])

    df = open_data()

    with tab_1:

        st.header("Исследуем наши 'чистые' данные!")

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

        st.subheader("Посмотрим на соотношение распределений по сумме взятого кредита между двумя категориями людей:"
                     "теми, кто откликнулся и теми, кто нет. ")
        st.pyplot(show_double_hist(df))
        st.markdown("Распределения по сумме взятого кредита примерно похожи для обеих категорий людей"
                " - которые откликнулись на рекламу и нет."
                " То есть сохраняется пропорциональность.")

        st.divider()

        st.subheader("Изучим числовые характеристики числовых признаков")
        st.dataframe(show_numerical_features(df))
        st.markdown("Cредний заработок примерно равен среднему значению взятого кредита;"
                "был человек(несколько людей) с 11 кредитами;"
                "огромная разница между минимальным и максимальным заработком(минимальный - 24, максимальный - 250,000)")

        st.divider()

        st.subheader("Изучим числовые характеристики категориальных признаков")
        st.write(explore_categorical(df))

    with tab_2:

        st.header("Попредсказываем!")

        with open('data/model.pickle', 'rb') as f:
            model = pickle.load(f)

        limit = st.slider("Выберите порог для перевода предсказаний модели в классы", 0.0, 1.0, 0.01)
        st.write(show_metrics(model, limit, df))
        st.divider()

        st.write("Давайте посмотрим, с какой вероятностью разные клиенты ответили бы на нашу рассылку!")
        option = st.selectbox('Выберите номер клиента', tuple([i for i in range(13855)]))
        X_train_transformed, X_val_transformed, y_train, y_val = make_matrices()
        df_new = pd.concat([X_train_transformed, X_val_transformed], axis=0)
        df_new.reset_index(inplace=True, drop=True)
        prob = model.predict_proba(df_new)[option][1]
        st.write(f"Данный клиент откликнется на маркетинговое предложение с вероятностью {round(prob, 2) * 100}%")









if __name__ == "__main__":
    show_main_page()
