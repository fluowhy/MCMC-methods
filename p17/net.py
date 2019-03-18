
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
    
class BatchingDataset():
    def __init__(self, datos, clases, C):
        self.C = C
        self.datos = datos
        self.N = len(self.datos)
        self.clases = torch.FloatTensor(np.eye(self.N, self.C)[clases])
        if torch.cuda.is_available():
            self.datos = self.datos.type(torch.cuda.FloatTensor)
            self.clases = self.clases.type(torch.cuda.FloatTensor)
        return

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        #print(self.Td[0])
        return self.datos[i]

    def paquetes(self, B):
        Tdb = []
        Tcb = []
        d = self.N%B
        if d==0:
            for i in range(0, self.N - 1, B):
                Tdb.append(self.datos[i:i + B])
                Tcb.append(self.clases[i:i + B])
        else:
            for i in range(0, self.N - d, B):
                Tdb.append(self.datos[i:i + B])
                Tcb.append(self.clases[i:i + B])
            Tdb.append(self.datos[i + B:])
            Tcb.append(self.clases[i + B:])
        return zip(Tdb, Tcb)
    
    
def train(net, dataset, optimizer, epochs, minibatches):
    criterion = nn.MSELoss()
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        i = 0
        for x, y in dataset.paquetes(minibatches):
            # get the inputs
            inputs = x
            labels = y
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = cross_ent_loss(outputs, y, epsilon=1e-10) # criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            i += 1
        print('Loss', running_loss/i)       
    print('Finished Training')
    return


def softmax(T, dim, estable=True):
    if not estable:
        m = torch.max(T, dim=dim, keepdim=True)[0].expand_as(T)
        T_m = T - m
        T_out = softmax(T_m, dim=dim, estable=True)
        return T_out
    elif estable:
        T_exp = torch.exp(T)   
        T_sum = torch.sum(T_exp, dim=dim, keepdim=True).expand_as(T)
        T_out = T_exp*torch.reciprocal(T_sum)
        T_out[T_out<1e-5] = 1e-5
        T_out[T_out>1 - 1e-5] = 1 - 1e-5
        if torch.cuda.is_available():
            T_out = T_out.type(torch.cuda.FloatTensor)
        return T_out
    

def cross_ent_loss(Q, P, epsilon=1e-10):
    N = Q.shape[0]
    Q[Q<epsilon] = epsilon
    Q[Q>1 - epsilon] = 1 - epsilon
    return - torch.sum(P.mul(torch.log(Q)))/N

