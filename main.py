import sys
from code.parameters import PARAMS, check_params

if __name__ == '__main__':

    if check_params(arg=sys.argv[1:]) == 0:
        exit(0)
    