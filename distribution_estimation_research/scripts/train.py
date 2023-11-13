import torch
import torch.nn as nn
from torch.optim import SGD
import numpy as np

def train_model(dl, f, n_epochs=20):
    # Optimization
    opt = SGD(f.parameters(), lr=0.01)
    L = nn.CrossEntropyLoss()

    # Train model
    losses = []
    epochs = []
    for epoch in range(n_epochs):
        print(f'Epoch {epoch}')
        N = len(dl)
        for i, (x, y) in enumerate(dl):
            # Update the weights of the network
            opt.zero_grad() 
            loss_value = L(f(x), y) 
            loss_value.backward() 
            opt.step() 
            epochs.append(epoch+i/N)
            losses.append(loss_value.item())
    return f, np.array(epochs), np.array(losses)

# Used to train a series of models
def train_models(train_dls, f_set, use_wandb):
    
    if use_wandb:
        import wandb
        
    epoch_data_set = []
    loss_data_set = []
    new_f_set = []
    for i in range(len(f_set)):
        print("Training model differentiating set " + str(i) + " and set " + str(i+1))
        f, epoch_data, loss_data = train_model(train_dls[i], f_set[i], 10)
        if use_wandb:
            wandb.log({f"model-{i} train loss": loss_data, "epoch": epoch_data})
        new_f_set.append(f)
        epoch_data_set.append(epoch_data)
        loss_data_set.append(loss_data)
    
    return new_f_set, epoch_data_set, loss_data_set