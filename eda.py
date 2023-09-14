from typing import Tuple, Any

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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


def show_double_hist(df: pd.DataFrame) -> plt.figure:
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.histplot(data=df, x=df.CREDIT, hue='TARGET', kde=True)

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


def make_matrices() -> tuple[DataFrame, DataFrame, Any, Any]:

    df = pd.read_csv("data/clients_clean.csv")

    X = df.drop(['TARGET', 'AGREEMENT_RK'], axis=1)
    y = df['TARGET']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

    categorical = ['FAMILY_INCOME', 'EDUCATION', 'MARITAL_STATUS']
    numeric_features = ['WORK_TIME', 'PERSONAL_INCOME', 'CREDIT', "TERM", "FST_PAYMENT",
                        "AGE", "GENDER", "CHILD_TOTAL", "DEPENDANTS", "SOCSTATUS_WORK_FL",
                        "SOCSTATUS_PENS_FL", "OWN_AUTO", "LOAN_NUM_TOTAL", "LOAN_NUM_CLOSED"]

    ct = ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown="ignore"), categorical),
        ('scaling', MinMaxScaler(), numeric_features)
    ])

    X_train_transformed = ct.fit_transform(X_train)
    X_val_transformed = ct.transform(X_val)

    new_features = list(ct.named_transformers_['ohe'].get_feature_names_out())
    new_features.extend(numeric_features)

    X_train_transformed = pd.DataFrame(X_train_transformed, columns=new_features)
    X_val_transformed = pd.DataFrame(X_val_transformed, columns=new_features)

    return X_train_transformed, X_val_transformed, y_train, y_val


def show_metrics(model, limit: float, df: pd.DataFrame) -> str:
    text = ""

    X_train_transformed, X_val_transformed, y_train, y_val = make_matrices()

    model.fit(X_train_transformed, y_train)
    pred_test = model.predict(X_val_transformed)

    probs = model.predict_proba(X_val_transformed)
    probs_churn = probs[:, 1]

    classes = probs_churn > limit

    text += f"\nAccuracy: {accuracy_score(y_val, classes)}\n"
    text += f"\nPrecision: {precision_score(y_val, classes)}\n"
    text += f"\nRecall: {recall_score(y_val, classes)}\n"
    text += f"\nf1: {f1_score(y_val, classes)}\n"

    return text



