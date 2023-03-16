import os, sys
from code.utils import colorify

DATA_FOLDER = 'data'


if __name__ == '__main__':
    
    # forder preparation
    if not os.path.isdir(DATA_FOLDER):
        os.mkdir(DATA_FOLDER)
        print ('# Created folder', colorify(DATA_FOLDER), 'please copy the data files there')