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

def train(
    model, optim, loss, X_train, y_train, 
    X_val, y_val, device, n_epoch=100, batch_size=100, 
    model_name='model', dtype_float=torch.float32
):
    save_dir = './output/'
    filepath = save_dir + model_name

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    # X_val = X_val.to(device)
    # y_val = y_val.to(device)

    scaler = torch.cuda.amp.GradScaler()

    t1 = time.time()

    for epoch in range(n_epoch):
        order = np.random.permutation(len(y_train))
        for batch in range(0, len(y_train), batch_size):
            optim.zero_grad()
            
            batch_id = order[batch:batch+batch_size]
            
            X_train_batch = X_train[batch_id].to(device)
            y_train_batch = y_train[batch_id].to(device)

            with torch.autocast(device_type='cuda', dtype=dtype_float):
                preds = model(X_train_batch)
                loss_val = loss(preds, y_train_batch)

            scaler.scale(loss_val).backward()
            scaler.step(optim)
            scaler.update()

        if epoch % 1 == 0:
            counter = 0
            accuracy = 0
            for batch in range(0, len(y_val), batch_size):
                X_val_batch = X_val[batch:batch+batch_size].to(device)
                y_val_batch = y_val[batch:batch+batch_size].to(device)

                with torch.autocast(device_type='cuda', dtype=dtype_float):
                    val_preds = model(X_val_batch)

                accuracy += (val_preds.argmax(dim=1) == y_val_batch.argmax(dim=1)).float().mean().data.cpu()
                counter += 1
            print('Validation accuracy = ', accuracy / counter)

    t2 = time.time()

    print('Total train time = ', t2 - t1)

    torch.save(model.state_dict(), filepath)
    
    return model