import torch
import numpy as np
import random
import torchvision.datasets

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

from train import *
from eval import *
from model import *
from utils import *

if __name__ == '__main__':

    mnist_train = torchvision.datasets.MNIST('./', download=True, train=True)
    mnist_test = torchvision.datasets.MNIST('./', download=True, train=False)

    X_train = mnist_train.data
    y_train = mnist_train.targets
    X_test = mnist_test.data
    y_test = mnist_test.targets

    X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
    X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min())

    X_train = X_train.unsqueeze(1).float()
    X_test = X_test.unsqueeze(1).float()

    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    train_split = 0.8
    train_size = int(train_split * X_train.shape[0])

    X_val = X_train[train_size:]
    y_val = y_train[train_size:]

    X_train = X_train[0:train_size]
    y_train = y_train[0:train_size]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = Model()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    loss = torch.nn.CrossEntropyLoss()


    train(model, optimizer, loss, 
            X_train, y_train, X_val, y_val,
            device, dtype_float=torch.float16, n_epoch=10)

    # eval()