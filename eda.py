import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def open_data(path: str = "data/clients_clean.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def explore_numerical(df: pd.DataFrame) -> plt.figure:
    fig, axs = plt.subplots(3, 3, figsize=(17,17))
    axs[0, 0].hist(df["PERSONAL_INCOME"], bins=50)
    axs[0, 0].set_title("Личный доход")
    axs[0, 1].hist(df["CREDIT"], bins=50)
    axs[0, 1].set_title("Сумма кредита")
    axs[0, 2].hist(df["TARGET"], bins=50)
    axs[0, 2].set_title("Целевая переменная")
    axs[1, 0].hist(df["AGE"], bins=50)
    axs[1, 0].set_title("Возраст")
    axs[1, 1].hist(df["GENDER"], bins=50)
    axs[1, 1].set_title("Пол")
    axs[1, 2].hist(df["FL_PRESENCE_FL"], bins=50)
    axs[1, 2].set_title("Квартира в собственности")
    axs[2, 0].hist(df["OWN_AUTO"], bins=50)
    axs[2, 0].set_title("Машина в собственности")
    axs[2, 1].hist(df["LOAN_NUM_TOTAL"], bins=50)
    axs[2, 1].set_title("Количество взятых кредитов")
    axs[2, 2].hist(df["LOAN_NUM_CLOSED"], bins=50)
    axs[2, 2].set_title("Количество закрытых кредитов")


    for ax in axs.flat:
        ax.set(ylabel='Количество')

    return fig


def show_correlation(df: pd.DataFrame) -> plt.figure:
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(df.corr(), ax=ax)

    return fig


def show_dependance_on_target(df: pd.DataFrame, name: str) -> plt.figure:
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.scatterplot(x=df[name], y=df.TARGET, ax=ax)

    return fig


def show_numerical_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe()


def explore_categorical(df: pd.DataFrame) -> str:
    text = ""
    text += "Данные по образованию  \n"
    for i in range(df.EDUCATION.nunique()):
        el = df.EDUCATION.unique()[i]
        text += f"{el}: {df.EDUCATION.value_counts()[el]}  \n"

    text += "  \n  \n  \n  Данные по доходу семьи  \n"
    for i in range(df.FAMILY_INCOME.nunique()):
        el = df.FAMILY_INCOME.unique()[i]
        text += f"{el}: {df.FAMILY_INCOME.value_counts()[el]}  \n"

    text += "  \n  \n  \n  Данные по семейному положению  \n"
    for i in range(df.MARITAL_STATUS.nunique()):
        el = df.MARITAL_STATUS.unique()[i]
        text += f"{el}: {df.MARITAL_STATUS.value_counts()[el]}  \n"

    return text