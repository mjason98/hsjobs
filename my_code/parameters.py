import argparse, os
from my_code.utils import colorify

PARAMS = {
    # general parameters
    'seed':1234567,

    # train parameters
    "lr": 0.001,
    'optim':'adam',
    'workers':2,
    'batch':12,
    'epochs':12,
    
    # model values
    'TRANS_NAME':'bert-base-uncased',
    'MODEL_FOLDER':'pts',
    
    # dataset values
    "DATA_FOLDER":"data",
    'DATA_PATH':'data/fake_job_postings.csv',
    'DATA_TARGET_COLUMN_NAME':'fraudulent',
    'data_train':'data/train.csv',
    'data_test':'data/test.csv',
    'data_percent':0.05,
    

    # ...
}

def check_params(arg=None):
    global PARAMS

    parse = argparse.ArgumentParser(description='Harbour Space Fake Job Post Detector')

    parse.add_argument('--datafolder', dest='datafolder', help='Data folder path', 
                       required=False, default='data')

    parse.add_argument('--lr', dest='lr', help='Learning rate value', 
                       required=False, default=0.001)
   
    returns = parse.parse_args(arg)
    new_params = {
        'DATA_FOLDER':returns.datafolder,
        'lr':returns.lr,
    }

    PARAMS.update(new_params)

    # forder preparation
    if not os.path.isdir(PARAMS['DATA_FOLDER']):
        os.mkdir(PARAMS['DATA_FOLDER'])
        print ('# Created folder', colorify(PARAMS['DATA_FOLDER']), 'please copy the data files there')
    
    if not os.path.isdir(PARAMS['MODEL_FOLDER']):
        os.mkdir(PARAMS['MODEL_FOLDER'])
        print ('# Created folder', colorify(PARAMS['MODEL_FOLDER']), 'to save the models weights')

    return 1