import torch
import numpy as np
import random

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        n_hiddin_neurons = 200
        
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=n_hiddin_neurons, kernel_size=5, padding=3)
        self.act1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = torch.nn.Conv2d(in_channels=n_hiddin_neurons, out_channels=n_hiddin_neurons, kernel_size=5, padding=3)
        self.act2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = torch.nn.Conv2d(in_channels=n_hiddin_neurons, out_channels=n_hiddin_neurons, kernel_size=5, padding=3)
        self.act3 = torch.nn.ReLU()
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = torch.nn.Linear(5000, n_hiddin_neurons)
        self.act3 = torch.nn.ReLU()
        
        self.fc2 = torch.nn.Linear(n_hiddin_neurons, 10)
        sm = torch.nn.Softmax()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.act3(x)
        x = self.pool3(x)
        
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        
        x = self.fc1(x)
        x = self.act3(x)
        x = self.fc2(x)
        
        return x