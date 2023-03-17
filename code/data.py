# data preprocesing funtions here

import pandas as pd
from random import sample
import re

from code.parameters import PARAMS


def load_data(data_path: str):
    data = pd.read_csv(data_path)
    return data

def save_data(data:pd.Series):
    file_path_train = PARAMS['data_train']
    file_path_test = PARAMS['data_test']
    target_name = PARAMS['DATA_TARGET_COLUMN_NAME']
    percent = PARAMS['data_percent']

    data_true  = data.query(target_name+'==1')
    data_false = data.query(target_name+'==0')

    train_true_ids = sample(data_true.index.tolist(), int(len(data_true)*percent))
    train_false_ids = sample(data_false.index.tolist(), int(len(data_false)*percent))

    data_train = pd.concat([ data_true.loc[ train_true_ids ], data_false.loc[ train_false_ids ] ], axis=1)
    data_train.to_csv(file_path_train, index=False)

    del data_train

    data_true = data_true.drop(train_true_ids)
    data_false = data_false.drop(train_false_ids)

    data_test = pd.concat([ data_true, data_false ], axis=1)
    data_test.to_csv(file_path_test, index=False)

    del data_test

    return file_path_train, file_path_test


def clean_data(df: pd.DataFrame):
    df[["country", "city_code", "city_name"]] = df["location"].str.split(
        ",", n=2, expand=True
    )

    df = df.drop(["salary_range", "location"], axis=1)

    string_columns = df.select_dtypes(include="object").columns.tolist()
    df = df[string_columns].fillna("This field is not specified")

    def fix_string_col(text: str):
        text = text.strip()
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

        return text

    for c in string_columns:
        df[c] = df[c].apply(fix_string_col)

    return df


# ----------------------------------------------------------------

DATA_PIPELINE = [load_data, clean_data, save_data]


def processData():
    curr_value = PARAMS["DATA_PATH"]

    for f in DATA_PIPELINE:
        curr_value = f(curr_value)
