# Harbour Space Fake Job Post Detector 

This project is part of an assignment given to students at Harbour Space University. It is intended to detect from job post descriptions, and some metadata, whether it is fake or not.

## Usage

### Environment 

Then create the environment and install the dependencies with pip.
```shell
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the project
To run the project, make sure you have a `.env` file in the root folder.

It should look exactly like `.env.example` but with your real TELEGRAM TOKEN.

Once you do it, you can run the main file, it will prepare the data and train the model:
```shell
python main.py
```

After it is done, run the chatbot by using:
```shell
python -m my_code.chatbot
```

#### Data Cleaning

1. Remove the salary range column
2. split location column into- country, state, city from location

3. Replace null to string "missing" -- instead of dropping missings, use as a valid observation. Could be that fake posts often have missing fields
4. Drop non-english text entries
5. Clean text columns: separate sentences, remove URLs, non-ascii, punctuation, extra spaces and white spac
6. combine text into single test, Tokenization, remove stopwords,other text cleaning and processing of data
7. vectorization


### Train

### Predict
