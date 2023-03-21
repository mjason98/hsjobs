# data preprocesing funtions here

import pickle
from random import sample
import re

import pandas as pd

from my_code.parameters import PARAMS


def load_data(data_path: str):
    data = pd.read_csv(data_path)
    return data


def save_data(data: pd.Series):
    file_path_train = PARAMS["data_train"]
    file_path_test = PARAMS["data_test"]
    target_name = PARAMS["DATA_TARGET_COLUMN_NAME"]
    percent = 1.0 - PARAMS["data_percent"]

    data_true = data.query(target_name + "==1")
    data_false = data.query(target_name + "==0")

    train_true_ids = sample(data_true.index.tolist(), int(len(data_true) * percent))
    train_false_ids = sample(data_false.index.tolist(), int(len(data_false) * percent))

    data_train = pd.concat(
        [data_true.loc[train_true_ids], data_false.loc[train_false_ids]], axis=0
    )
    data_train.to_csv(file_path_train, index=False)

    del data_train

    data_true = data_true.drop(train_true_ids)
    data_false = data_false.drop(train_false_ids)

    data_test = pd.concat([data_true, data_false], axis=0)
    data_test.to_csv(file_path_test, index=False)

    del data_test

    return file_path_train, file_path_test


def clean_data(df: pd.DataFrame):
    df[["country", "city_code", "city_name"]] = df["location"].str.split(
        ",", n=2, expand=True
    )

    df = df.drop(["salary_range", "location"], axis=1)

    def fix_string_col(text: str):
        text = text.encode("ascii", "ignore").decode()  # ignore non ascii characters

        text = re.sub(
            "([A-Z])((?=[a-z]))", r" \1", text
        )  # if lower case followed by upper case, separate by space

        text = re.sub("http[^\s]+ ", " ", text)
        text = re.sub("url[^\s]+ ", " ", text)
        text = re.sub(
            r"[^\w\s]", "", text
        )  # remove punctuation. Replace with '' so don't separate contractions
        text = re.sub(" +", " ", text)  # remove double and triple spaces
        text = text.strip()

        return text

    string_columns = df.select_dtypes(include="object").columns
    df[string_columns] = df[string_columns].fillna("")

    for c in string_columns:
        df[c] = df[c].apply(fix_string_col)

    unspecified = "This field is not specified"
    df[string_columns] = df[string_columns].replace("", unspecified)

    # there are some columns that have numbers
    df[string_columns] = df[string_columns].replace(r"^\d+$", unspecified, regex=True)

    return df


def vectorization(df: pd.DataFrame):
    print ('# Data vecorization phase')
    unspecified = "This field is not specified"

    def binary(x):
        return int(x != unspecified)

    to_binary = ["country", "city_name", "city_code"]

    for feat in to_binary:
        df["valid_" + feat] = df[feat].apply(binary)

    df = df.drop(to_binary, axis=1)

    categoricals = [
        "employment_type",
        "required_experience",
        "required_education",
        "industry",
        "function",
    ]

    cat_vector = pd.get_dummies(df[categoricals])

    pos_from_cat = {}
    for i, x in enumerate(cat_vector.columns):
        pos_from_cat[x] = i

    with open(PARAMS["cat_vector"], "wb") as fd:
        pickle.dump(pos_from_cat, fd)

    df["cat_vector"] = cat_vector.values.tolist()
    df["cat_vector"] = df["cat_vector"].apply(lambda x: " ".join(map(str, x)))
    df = df.drop(categoricals, axis=1)

    return df


def vect_from_dict(input: dict):
    with open(PARAMS["cat_vector"], "rb") as fd:
        pos_from_cat = pickle.load(fd)

    v = [0] * len(pos_from_cat)
    for cat, value in input.items():
        key = f"{cat}_{value}"
        assert key in pos_from_cat
        v[pos_from_cat[key]] = 1

    return " ".join(map(str, v))


# ----------------------------------------------------------------

DATA_PIPELINE = [load_data, clean_data, vectorization, save_data]


def processData():
    curr_value = PARAMS["DATA_PATH"]

    for f in DATA_PIPELINE:
        curr_value = f(curr_value)
