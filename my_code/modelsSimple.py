from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd, numpy as np
from my_code.parameters import PARAMS
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib, os

def setSeed():
    my_seed = PARAMS['seed']
    np.random.seed(my_seed)
    random.seed(my_seed)

def loadData():
    data_train = pd.read_csv(PARAMS['data_train'])
    data_test = pd.read_csv(PARAMS['data_test'])

    return data_train, data_test

def trainModel():
    print ('# Training Model')

    task = PARAMS['DATA_TARGET_COLUMN_NAME']

    train, test = loadData()
    vectorizerTitle = TfidfVectorizer(min_df = 0, 
                                max_df = 0.8, 
                                sublinear_tf = True,
                                analyzer = 'char',
                                ngram_range=(3, 3), 
                                use_idf = True)

    vectorizerDescription = TfidfVectorizer(min_df = 0, 
                                max_df = 0.8, 
                                sublinear_tf = True,
                                analyzer = 'char',
                                ngram_range=(3, 3), 
                                use_idf = True)

    # titleTrain = vectorizerTitle.fit_transform(train['title']).toarray()
    # titleTest = vectorizerTitle.transform(test['title']).toarray()

    # trainVec = np.array(list(map(lambda x: [int(v) for v in x.split(' ')], train['cat_vector']  )))
    # testVec = np.array(list(map(lambda x: [int(v) for v in x.split(' ')], test['cat_vector']  )))

    descriptionTrain = vectorizerDescription.fit_transform(train['description']).toarray()
    descriptionTest = vectorizerDescription.transform(test['description']).toarray()

    # trainV = np.concatenate([trainVec, descriptionTrain], axis=1)
    # testV = np.concatenate([testVec, descriptionTest], axis=1)

    # del titleTrain
    # del titleTest
    # del trainVec
    # del testVec
    # del descriptionTrain
    # del descriptionTest

    model = RandomForestClassifier()

    model.fit(descriptionTrain, train[task])
    pred = model.predict(descriptionTest)

    metrics = classification_report(test[task], pred, target_names=[f'No {task}', task],  digits=4, zero_division=1)        
    print(metrics)

    # save models
    joblib.dump(vectorizerDescription, os.path.join(PARAMS['MODEL_FOLDER'], 'text2vec.joblib'))
    joblib.dump(model, os.path.join(PARAMS['MODEL_FOLDER'], 'randomForest.joblib'))


def predict(values:dict, model=None):
    '''
        Use the folowwing values:

        values: {
            "title": "the title",
            "description": "the description",
            "department": "",
            ""
        }
    '''
    vectorizerDescription = joblib.load(os.path.join(PARAMS['MODEL_FOLDER'], 'text2vec.joblib'))

    if model is None:
        model = joblib.load(os.path.join(PARAMS['MODEL_FOLDER'], 'randomForest.joblib'))

    text = values['description']

    descriptionTest = vectorizerDescription.transform([text])
    pred = model.predict(descriptionTest)

    return pred 

