import argparse, os
from code.utils import colorify

PARAMS = {
    "lr": 0.001,
    "DATA_FOLDER":"data",
    'seed':1234567,
    'TRANS_NAME':'',
    'PROCESED_DATA_PATH':'data/tmp.csv',
    'DATA_PATH':'data/fake_job_postings.csv',
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

    return 1
