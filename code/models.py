# models here

import pandas as pd, numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

from code.parameters import PARAMS

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

        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
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
        
        # ids = int(self.reg.sub("", "0" + str(self.data_frame.loc[idx, self.id_name])))
        ids = int(self.data_frame.loc[idx, self.id_name])
        
        # text fields
        sent1 = 'Title: ' + self.data_frame.loc[idx, self.x1]
        sent2 = 'Description:' + self.data_frame.loc[idx, self.x2]

        sent = ' '.join([sent1, sent2])

        # target field
        target = int(self.data_frame.loc[idx, self.y_name])

        sample = {'x': sent, 'y': target, 'id':ids}
        return sample

def makeDataSet(csv_path:str, batch, shuffle=True):
    data   =  mydataset(csv_path)
    loader =  DataLoader(data, batch_size=batch, shuffle=shuffle, num_workers=PARAMS['workers'], drop_last=False)
    return data, loader

def trainModel():
    model = Encoder_Model(500)
    optim = model.makeOptimizer(lr=PARAMS['lr'], algorithm=PARAMS['optim'])

    _, data_train_l = makeDataSet(PARAMS['data_train'], PARAMS['batch'])
    _, data_test_l = makeDataSet(PARAMS['data_test'], PARAMS['batch'])

    model.train()
    epochs = PARAMS['epochs']

    for e in range(epochs):
        total_loss, total_acc, dl = 0., 0., 0
        for data in data_train_l:
            optim.zero_grad()
            
            y_hat = model(data['x'])
            y1    = data['y'].to(device=model.device).flatten() if not use_reg else data['v'].to(device=model.device).flatten()
            try:
                loss = model.criterion1(y_hat, y1)
            except:
                # tamano 1
                y_hat = y_hat.view(1,-1)
                loss  = model.criterion1(y_hat, y1)
        
            loss.backward()
            optim.step()

            with torch.no_grad():
                total_loss += loss.item() * y1.shape[0]
                if use_acc: 
                    total_acc += (y1 == y_hat.argmax(dim=-1).flatten()).sum().item()
                    if mtl:
                        total_mse += l2.item() * y2.shape[0]
                        total_acc_2 += (y3 == y_mec.argmax(dim=-1).flatten()).sum().item()
                        total_acc_3 += (y4.flatten() == ( torch.sigmoid(y_tar) > 0.5 ).flatten()).sum().item() / y_tar.shape[-1]
                dl += y1.shape[0]
            bar.next(total_loss/dl)
        if use_acc:
            res = board.update('train', total_acc/dl, getBest=True)
            if mtl:
                res2 = board.update('train_mse', total_mse/dl, getBest=True)
                res3 = board.update('train_acc2', total_acc_2/dl, getBest=True)
                res4 = board.update('train_acc3', total_acc_3/dl, getBest=True)
                if bett == 'reg1':
                    res = res2
                elif bett == 'class2':
                    res = res3
                elif bett == 'class3':
                    res = res4
        else:
            res = board.update('train', total_loss/dl, getBest=True)
        
        # Evaluate the model
        if evalData_loader is not None:
            total_loss, total_acc, dl= 0,0,0
            if mtl:
                total_mse = 0.0
                total_acc_2 = 0.0
                total_acc_3 = 0.0
            with torch.no_grad():
                for data in evalData_loader:
                    if mtl:
                        y_hat, y_val, y_mec, y_tar = model(data['x'])
                        # y_val.float
                        y1 = data['y'].to(device=model.device)
                        y2 = data['v'].to(device=model.device).float()
                        y3 = data['m'].to(device=model.device)
                        y4 = data['t'].to(device=model.device)
                        # Tamano 1
                        try:
                            l1 = model.criterion1(y_hat, y1)
                            l2 = model.criterion2(y_val, y2, y1)
                            l3 = model.criterion3(y_mec, y3)
                            l4 = model.criterion4(y_tar, y4)
                            loss = etha[0]*l1 + etha[1]*l2 + etha[2]*l3 + etha[3]*l4
                        except:
                            y_hat = y_hat.view(1,-1)
                            l1 = model.criterion1(y_hat, y1)
                            l2 = model.criterion2(y_val, y2, y1)
                            l3 = model.criterion3(y_mec, y3)
                            l4 = model.criterion4(y_tar, y4)
                            loss = etha[0]*l1 + etha[1]*l2 + etha[2]*l3 + etha[3]*l4
                    else:
                        y_hat = model(data['x'])
                        y1    = data['y'].to(device=model.device).flatten() if not use_reg else data['v'].to(device=model.device).flatten()
                        loss = model.criterion1(y_hat, y1)
                    
                    total_loss += loss.item() * y1.shape[0]
                    if use_acc:
                        total_acc += (y1 == y_hat.argmax(dim=-1)).sum().item()
                        if mtl:
                            total_mse += l1.item() * y2.shape[0]
                            total_acc_2 += (y3 == y_mec.argmax(dim=-1).flatten()).sum().item()
                            total_acc_3 += (y4.flatten() == ( torch.sigmoid(y_tar) > 0.5 ).flatten()).sum().item() / y_tar.shape[-1]
                    dl += y1.shape[0]
                    bar.next()
            if use_acc:
                res = board.update('test', total_acc/dl, getBest=True)
                if mtl:
                    res2 = board.update('test_mse', total_mse/dl, getBest=True)
                    res3 = board.update('test_acc2', total_acc_2/dl, getBest=True)
                    res4 = board.update('test_acc3', total_acc_3/dl, getBest=True)
                    if bett == 'reg1':
                        res = res2
                    elif bett == 'class2':
                        res = res3
                    elif bett == 'class3':
                        res = res4
            else:
                res = board.update('test', total_loss/dl, getBest=True)
        bar.finish()
        del bar
        
        if res:
            model.save(os.path.join('pts', nameu+'.pt'))
    # board.show(os.path.join('out', nameu+'.png'), plot_smood=smood, pk_save=True)

