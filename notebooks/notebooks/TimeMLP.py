import numpy as np
import pylab as plt

import torch
import torch.nn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import time

from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder


class MNISTdataset(torch.utils.data.Dataset):

    def __init__(self):

        d0 = np.fromfile('mnist-data/data0', dtype=np.uint8).reshape((1000, 28, 28))
        d1 = np.fromfile('mnist-data/data1', dtype=np.uint8).reshape((1000, 28, 28))
        d2 = np.fromfile('mnist-data/data2', dtype=np.uint8).reshape((1000, 28, 28))
        d3 = np.fromfile('mnist-data/data3', dtype=np.uint8).reshape((1000, 28, 28))
        d4 = np.fromfile('mnist-data/data4', dtype=np.uint8).reshape((1000, 28, 28))
        d5 = np.fromfile('mnist-data/data5', dtype=np.uint8).reshape((1000, 28, 28))
        d6 = np.fromfile('mnist-data/data6', dtype=np.uint8).reshape((1000, 28, 28))
        d7 = np.fromfile('mnist-data/data7', dtype=np.uint8).reshape((1000, 28, 28))
        d8 = np.fromfile('mnist-data/data8', dtype=np.uint8).reshape((1000, 28, 28))
        d9 = np.fromfile('mnist-data/data9', dtype=np.uint8).reshape((1000, 28, 28))
        dataX = np.concatenate((d0, d1, d2, d3, d4, d5, d6, d7, d8, d9))
        dataY = np.concatenate(([0]*1000, [1]*1000, [2]*1000, [3]*1000, [4]*1000, [5]*1000, [6]*1000, [7]*1000, [8]*1000, [9]*1000))
        
        indices = np.random.permutation(10000)
        dataX = dataX[indices]
        dataY = dataY[indices]
        self.dataX = dataX.reshape(10000, 784).astype(np.float32) / 255
        self.dataY = dataY
#         self.dataY = OneHotEncoder(categories='auto', sparse=False).fit_transform(dataY.reshape(-1, 1))

    def __len__(self):
        return self.dataX.shape[0]

    def __getitem__(self, i):
        img = torch.Tensor(self.dataX[i])
        num = self.dataY[i]
        return (img, num)
    
class Net(torch.nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.d1 = torch.nn.Linear(784, 512)
        self.d2 = torch.nn.Linear(512, 512)
        self.d3 = torch.nn.Linear(512, 10)
        
    def forward(self, x):
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        x = F.softmax(self.d3(x))
        return x
    
model = Net()

dataset = MNISTdataset()
traingenerator = torch.utils.data.DataLoader(dataset=dataset, batch_size=1024, shuffle=True)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.0)

losses = np.zeros(20)
for i in tqdm(range(20)):
    for batch in traingenerator:
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = batch
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(inputs.float())
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        for batch in traingenerator:
            inputs, labels = batch
            outputs = model(inputs.float())
            loss = criterion(outputs, labels.long())
            losses[i] += loss.item()

dataset = MNISTdataset()
yt = dataset.dataY
yp = np.zeros(yt.shape[0])
for i in tqdm(range(yp.shape[0])):
    x = torch.tensor(dataset.dataX[i][np.newaxis, :])
    y = model(x)
    yp[i] = y.detach().numpy().argmax(axis=1)
# yp = ypp.argmax(axis=0)
print(classification_report(yt, yp))