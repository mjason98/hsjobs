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


#### DATA CLEANING

1. Remove the salary range column
2. split location column into- country, state, city from location

3. Replace null to string "missing" -- instead of dropping missings, use as a valid observation. Could be that fake posts often have missing fields
4. Drop non-english text entries
5. Clean text columns: separate sentences, remove URLs, non-ascii, punctuation, extra spaces and white spac
6. combine text into single test, Tokenization, remove stopwords,other text cleaning and processing of data
7. vectorization


### Train

### Predict
