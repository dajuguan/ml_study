#!/usr/bin/env python
# coding: utf-8

## input: (1), output:(1), EWC continual learning
# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from model.ewc_regression import ElasticWeightConsolidation
import numpy as np


# In[2]:


from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
class CustomDataset(Dataset):
    def __init__(self):
        x = np.linspace(1,10,200, dtype="float64") 
        y = x**2 + 1
        x = x.reshape(len(x),1)
        y = y.reshape(len(y),1)
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return x, y


# In[3]:


mnist_train = CustomDataset()
mnist_test = CustomDataset()
train_loader = DataLoader(mnist_train, batch_size = 200, shuffle=False)
test_loader = DataLoader(mnist_test, batch_size = 100, shuffle=False)


# In[4]:


class LinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim, act='relu', use_bn=False):
        super(LinearLayer, self).__init__()
        self.use_bn = use_bn
        self.lin = nn.Linear(input_dim, output_dim)
        self.act = nn.ReLU() if act == 'relu' else act
        if use_bn:
            self.bn = nn.BatchNorm1d(output_dim)
    def forward(self, x):
        if self.use_bn:
            return self.bn(self.act(self.lin(x)))
        return self.act(self.lin(x))

class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.shape[0], -1)


# In[5]:


class BaseModel(nn.Module):
    
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super(BaseModel, self).__init__()
        self.f1 = Flatten()
        self.lin1 = LinearLayer(num_inputs, num_hidden, use_bn=False)
        self.lin2 = LinearLayer(num_hidden, num_hidden, use_bn=False)
        self.lin3 = nn.Linear(num_hidden, num_outputs)
        
    def forward(self, x):
        return self.lin3(self.lin2(self.lin1(self.f1(x))))


# In[6]:


crit = nn.MSELoss(reduce="sum")
ewc = ElasticWeightConsolidation(BaseModel(1, 200, 1), crit=crit, lr=1e-3)
ewc.model.double()

STEPS = 4000
# scheduler = torch.optim.lr_scheduler.MultiStepLR(ewc.optimizer, milestones=[0.5 * STEPS, 0.75 * STEPS], gamma=0.1)
for _ in range(STEPS):
    for input, target in train_loader:
        ewc.forward_backward_update(input, target)
#         scheduler.step()
        


# In[9]:


model = ewc.model.eval()
input = torch.tensor([3],dtype=torch.float64)
print(model(input))
input = torch.tensor([25],dtype=torch.float64)
print(25**2 +1)
print(model(input))

ewc.model.train()


# In[10]:


ewc.register_ewc_params(mnist_train, 200, 1)

ewc.model.eval()
input = torch.tensor([3],dtype=torch.float64)
print(model(input))
input = torch.tensor([25],dtype=torch.float64)
print(25**2 +1)
print(model(input))

ewc.model.train()


# In[11]:
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
class CustomDataset(Dataset):
    def __init__(self):
        x = np.linspace(20,30,200, dtype="float64") 
        y = x**2 + 1
        x = x.reshape(len(x),1)
        y = y.reshape(len(y),1)
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return x, y
mnist_train = CustomDataset()
mnist_test = CustomDataset()
train_loader = DataLoader(mnist_train, batch_size = 200, shuffle=False)
test_loader = DataLoader(mnist_test, batch_size = 100, shuffle=False)


# In[11]:
# scheduler = torch.optim.lr_scheduler.MultiStepLR(ewc.optimizer, milestones=[0.5 * STEPS, 0.75 * STEPS], gamma=0.1)
for _ in range(3500):
    for input, target in train_loader:
        ewc.forward_backward_update(input, target)
        # scheduler.step()


# In[12]:
model = ewc.model.eval()
input = torch.tensor([3],dtype=torch.float64)
print(model(input))
input = torch.tensor([25],dtype=torch.float64)
print(model(input))
print("after==============")

ewc.model.train()
ewc.register_ewc_params(mnist_train, 200, 1)
model = ewc.model.eval()
input = torch.tensor([3],dtype=torch.float64)
print(model(input))
input = torch.tensor([25],dtype=torch.float64)
print(model(input))



