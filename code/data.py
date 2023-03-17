# data preprocesing funtions here

import pandas as pd

from code.parameters import PARAMS

def load_data(data_path:str):
    data = pd.read_csv(data_path)
    return data

def save_data(data:pd.Series):
    file_path = PARAMS['PROCESED_DATA_PATH']
    data.to_csv(file_path)
    return file_path

# ----------------------------------------------------------------

DATA_PIPELINE = [load_data, save_data]

def processData():
    curr_value = PARAMS['DATA_PATH']

    for f in DATA_PIPELINE:
        curr_value = f(curr_value)
    