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
    model, X_test, y_test, 
    device, batch_size=100, 
    dtype_float=torch.float32
):
    t1 = time.time()

    counter = 0
    accuracy = 0
    for batch in range(0, len(y_test), batch_size):
        X_val_batch = X_test[batch:batch+batch_size].to(device)
        y_val_batch = y_test[batch:batch+batch_size].to(device)

        with torch.autocast(device_type='cuda', dtype=dtype_float):
            test_preds = model(X_val_batch)

        accuracy += (test_preds.argmax(dim=1) == y_val_batch.argmax(dim=1)).float().mean().data.cpu()
        counter += 1
    
    t2 = time.time()

    print('Test accuracy = ', accuracy / counter)
    print('Total eval time = ', t2 - t1)
    
    return model