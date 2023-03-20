# models here

import os
from tqdm import tqdm
import pandas as pd, numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import random
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel

from sklearn.metrics import precision_recall_fscore_support as prf

from my_code.parameters import PARAMS

def setSeed():
    my_seed = PARAMS['seed']
    torch.manual_seed(my_seed)
    np.random.seed(my_seed)
    random.seed(my_seed)

# function that creates the transformer and tokenizer for later uses
def make_trans_pretrained_model(mod_only=False):
    '''
        This function return (tokenizer, model)
    '''
    tokenizer, model = None, None
    
    tokenizer = AutoTokenizer.from_pretrained(PARAMS['TRANS_NAME'])
    model = AutoModel.from_pretrained(PARAMS['TRANS_NAME'])

    if mod_only:
        return model 
    else:
        return tokenizer, model

# Select a specific vector from a sequence
class POS(torch.nn.Module):
    def __init__(self, _p = 0):
        super(POS, self).__init__()
        self._p = _p
    def forward(self, X):
        return X[:,self._p]

# The encoder used in this work
class Encoder_Model(nn.Module):
    def __init__(self, hidden_size, vec_size=768, max_length=120, selection='first', mtl=False):
        super(Encoder_Model, self).__init__()
        self.criterion1 = nn.CrossEntropyLoss()

        self.max_length = max_length
        self.tok, self.bert = make_trans_pretrained_model()

        self.selection = POS(0)

        self.encoder_last_layer = nn.Linear(vec_size, 2)

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            
        self.to(device=self.device)
        
    def forward(self, X, ret_vec=False):
        ids   = self.tok(X, return_tensors='pt', truncation=True, padding=True, max_length=self.max_length).to(device=self.device)
        out   = self.bert(**ids)
        vects = self.selection(out[0])
        return self.encoder_last_layer(vects)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))

    def save(self, path):
        torch.save(self.state_dict(), path) 
    
    def makeOptimizer(self, lr=5e-5, lr_factor=9/10, decay=2e-5, algorithm='adam'):
        pars = [{'params':self.encoder_last_layer.parameters()}]

        for l in self.bert.encoder.layer:
            lr *= lr_factor
            D = {'params':l.parameters(), 'lr':lr}
            pars.append(D)
        try:
            lr *= lr_factor
            D = {'params':self.bert.pooler.parameters(), 'lr':lr}
            pars.append(D)
        except:
            print('#Warning: Pooler layer not found')

        if algorithm == 'adam':
            return torch.optim.Adam(pars, lr=lr, weight_decay=decay)
        elif algorithm == 'rms':
            return torch.optim.RMSprop(pars, lr=lr, weight_decay=decay)

class mydataset(Dataset):
    def __init__(self, csv_file):
        self.data_frame = pd.read_csv(csv_file)
        self.x1  = 'title'
        self.x2  = 'description'
        self.id_name = 'job_id'
        self.y_name = 'fraudulent'

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ids = int(self.data_frame.loc[idx, self.id_name])
        
        # text fields
        sent1 = 'title: ' + self.data_frame.loc[idx, self.x1]
        sent2 = 'description:' + self.data_frame.loc[idx, self.x2]

        sent = ' '.join([sent1, sent2])

        # target field
        target = int(self.data_frame.loc[idx, self.y_name])

        sample = {'x': sent, 'y': target, 'id':ids}
        return sample

def getWeights(csv_path:str):
    data = pd.read_csv(csv_path)
    target_name = PARAMS["DATA_TARGET_COLUMN_NAME"]

    positive = len(data.query(target_name + "==1"))
    negative = len(data.query(target_name + "==0"))

    wc = [negative, positive]

    weights = [ 1/wc[ int( data.loc[i, target_name] ) ] for i in range(len(data)) ]

    return weights


def makeDataSet(csv_path:str, batch, shuffle=True, balanse=False):
    data   =  mydataset(csv_path)

    sampler = None
    if balanse:
        sample_weight = getWeights(csv_path)
        sampler = WeightedRandomSampler(weights=sample_weight, num_samples=len(data), replacement=True)
        shuffle = None

    loader =  DataLoader(data, batch_size=batch, shuffle=shuffle, num_workers=PARAMS['workers'], drop_last=False, sampler=sampler)
    return data, loader

def trainModel():
    model_path = os.path.join(PARAMS['MODEL_FOLDER'], 'model.pt')

    model = Encoder_Model(500)
    optim = model.makeOptimizer(lr=PARAMS['lr'], algorithm=PARAMS['optim'])

    _, data_train_l = makeDataSet(PARAMS['data_train'], PARAMS['batch'], balanse=True)
    _, data_test_l = makeDataSet(PARAMS['data_test'], PARAMS['batch'])

    dataloaders = {'train': data_train_l, 'val':data_test_l}

    epochs = PARAMS['epochs']

    model.save(model_path)

    for e in range(epochs):
        

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            iter = tqdm(dataloaders[phase])

            total_loss, total_acc, dl, best_acc = 0., 0., 0, 0.
            
            for data in iter:
                optim.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    y_hat = model(data['x'])
                    y1    = data['y'].to(device=model.device).flatten()

                    loss = model.criterion1(y_hat, y1)
                    
                    if phase == 'train':
                        loss.backward()
                        optim.step()

                    total_loss += loss.item() * y1.shape[0]
                    total_acc += (y1 == y_hat.argmax(dim=-1).flatten()).sum().item()
                    dl += y1.shape[0]
            
            if best_acc < total_acc and phase == 'val':
                best_acc = total_acc
                model.save(model_path)

            print('# {} epoch {} Loss {:.3} Acc {:.3}{}'.format(phase, e, total_loss/dl, total_acc/dl, '*' if total_acc == best_acc else ' '))

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
    if model is None:
        model_path = os.path.join(PARAMS['MODEL_FOLDER'], 'model.pt')
        model = Encoder_Model(500)
        model.load(model_path)
    
    model.eval()

    text = ' '.join(['title: ' + values['title'], 'description: ' + values['description']])

    with torch.no_grad():
        y_hat = model([text])
        pred = y_hat.argmax(dim=-1).flatten().cpu().numpy().tolist()
        return pred

def predictTest(model=None):
    print ('# Making predictions over test dataset')

    _, data_loader = makeDataSet(PARAMS['data_test'], PARAMS['batch'], shuffle=False)
    
    if model is None:
        model_path = os.path.join(PARAMS['MODEL_FOLDER'], 'model.pt')
        model = Encoder_Model(500)
        model.load(model_path)

    preds = []

    total_loss, total_acc, dl = 0., 0., 0
    
    iter = tqdm(data_loader)
    for data in iter:
        with torch.no_grad():
            y_hat = model(data['x'])
            y1    = data['y'].to(device=model.device).flatten()

            loss = model.criterion1(y_hat, y1)

            total_loss += loss.item() * y1.shape[0]
            total_acc += (y1 == y_hat.argmax(dim=-1).flatten()).sum().item()
            dl += y1.shape[0]

            preds.append(y_hat.argmax(dim=-1).cpu().numpy())
    
    preds = np.concatenate(preds, axis=0)
    
    print ('# Total loss {:.3} ACC {:.3}'.format(total_loss/dl, total_acc/dl))

    return preds

def showModelMetrics():
    y_hat = predictTest()
    y = pd.read_csv(PARAMS['data_test'])[PARAMS["DATA_TARGET_COLUMN_NAME"]]

    metrics = prf(y, y_hat)

    print ("# precision\t",   metrics[0])
    print ("# recall\t",      metrics[1])
    print ("# fbeta_score\t", metrics[2])
    print ("# support\t",     metrics[3])
    
