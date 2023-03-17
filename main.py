import sys
from code.parameters import check_params
from code.data import processData
from code.models import setSeed

if __name__ == '__main__':

    # PARAMETERS
    if check_params(arg=sys.argv[1:]) == 0:
        exit(0)

    # Seed
    setSeed()
    
    # DATA PIPELINE
    processData()
