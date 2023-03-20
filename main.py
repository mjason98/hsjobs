import sys
from my_code.parameters import check_params
from my_code.data import processData
from my_code.models import setSeed, trainModel, predictSingleText
from subprocess import call

if __name__ == '__main__':

    # PARAMETERS
    if check_params(arg=sys.argv[1:]) == 0:
        exit(0)

    # Seed
    setSeed()
    
    # DATA PIPELINE
    processData()

    # Model train 
    # trainModel()

    # todo: temporal function here

    predictSingleText('Title: Hi Description: this is a dummy test')

    # run the telegram bot
    #call(["python", "my_code/chatbot.py"])
    