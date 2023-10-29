import os

import torch
import numpy as np
import random
import time

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

def eval(
    model, loss, X_test, y_test, 
    device, batch_size=100, 
    dtype_float=torch.float16
):

    X_test = X_test.to(device)
    y_test = y_test.to(device)

    scaler = torch.cuda.GradScaler()

    t1 = time.time()

    with torch.cuda.amp.autocast(device_type=device, dtype=dtype_float):
        preds = model(X_test)

    accuracy = (preds.argmax(dim=1) == y_test).float().mean().data.cpu()
    print('Test accuracy = ', accuracy)

    t2 = time.time()

    print('Total eval time = ', t2 - t1)
    
    return model