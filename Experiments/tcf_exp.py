from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Solar, Dataset_PEMS
from torch.utils.data import DataLoader

from model.tcf import Model

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

import random
import os



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(2021)

size = [96,48,96]

train_set = Dataset_ETT_minute(
    root_path='C:/Users/Awsftausif/Desktop/S-Mamba_datasets/ETT-small/',
    data_path='ETTm1.csv',
    flag='train',
    size=size,
    features='M',      # 'M' = multivariate (use all features)
    target='OT',  # change this to your target column
    scale=True,
    timeenc=0,
    freq='t'           # depends on your dataset frequency (h=hourly, d=daily, etc.)
)

val_set = Dataset_ETT_minute(
    root_path='C:/Users/Awsftausif/Desktop/S-Mamba_datasets/ETT-small/',
    data_path='ETTm1.csv',
    flag='val',
    size=size,
    features='M',
    target='OT',
    scale=True,
    timeenc=0,
    freq='t'
)

test_set = Dataset_ETT_minute(
    root_path='C:/Users/Awsftausif/Desktop/S-Mamba_datasets/ETT-small/',
    data_path='ETTm1.csv',
    flag='test',
    size=size,
    features='M',
    target='OT',
    scale=True,
    timeenc=0,
    freq='t'
)


train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)


print(len(train_loader))
print(len(val_loader))
print(len(test_loader))

class Config:
    def __init__(self):
        self.d_model = 128
        self.d_core = 64
        self.e_layers = 2
        self.d_ff = 128
        self.n_vars = 7
        self.seq_len = 96
        self.pred_len = size[2]
        self.kernel_size = 0
        self.patch_len = 16
        self.n_heads = 8
        self.factor = 3       
        self.dropout = 0.1
        self.use_norm = True

# Create configuration instance
configs = Config()
model = Model(configs).to('cuda')



# ---- Loss & Metrics ----
mse_loss = nn.MSELoss()

def mae_loss(pred, true):
    return torch.mean(torch.abs(pred - true))

# ---- Train & Evaluate ----
def train(model, train_loader, optimizer, device, pred_len):
    model.train()
    total_loss = []
    for seq_x, seq_y, seq_x_mark, seq_y_mark in tqdm(train_loader, desc="Training", leave=False):
        optimizer.zero_grad()
        seq_x = seq_x.to(device).float()
        target = seq_y[:, -pred_len:, :].to(device).float()  # take last pred_len steps

        
        outputs = model(seq_x)
        loss = mse_loss(outputs, target)
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())
        

    return np.average(total_loss)


def evaluate(model, val_loader, device, pred_len):
    model.eval()
    total_mse, total_mae = 0, 0
    total_loss = []
    with torch.no_grad():
        for seq_x, seq_y, seq_x_mark, seq_y_mark in tqdm(val_loader, desc="Validating", leave=False):
            seq_x = seq_x.to(device).float()
            target = seq_y[:, -pred_len:, :].to(device).float()

            outputs = model(seq_x)
            loss = mse_loss(outputs, target)
            total_loss.append(loss.item())
            total_mae += mae_loss(outputs, target).item()

    return np.average(total_loss), total_mae / len(val_loader)


def test(model, test_loader, device, pred_len):
    model.eval()
    total_mse, total_mae = 0, 0
    with torch.no_grad():
        for seq_x, seq_y, seq_x_mark, seq_y_mark in tqdm(test_loader, desc="Testing", leave=False):
            seq_x = seq_x.to(device).float()
            target = seq_y[:, -pred_len:, :].to(device).float()

            outputs = model(seq_x)
            total_mse += mse_loss(outputs, target).item()
            total_mae += mae_loss(outputs, target).item()
    return total_mse / len(test_loader), total_mae / len(test_loader), None, None



# ---- Main Training Loop ----
def train_model(model, train_loader, val_loader, test_loader, pred_len, epochs=40, lr=0.0001, patience=5, device='cuda'):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    
    best_val_loss = float('inf')
    best_model = None
    patience_counter = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        train_loss = train(model, train_loader, optimizer, device, pred_len)
        val_mse, val_mae = evaluate(model, val_loader, device, pred_len)
        scheduler.step()
        print(f"Learning rate after epoch {epoch+1}: {scheduler.get_last_lr()[0]:.8f}")

        print(f"Train Loss: {train_loss:.7f} | Val MSE: {val_mse:.7f} | Val MAE: {val_mae:.7f}")

        # Early stopping
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Load best model
    model.load_state_dict(best_model)

    # Final test
    test_mse, test_mae, preds, trues = test(model, test_loader, device, pred_len)
    print(f"\nTest MSE: {test_mse:.6f} | Test MAE: {test_mae:.6f}")
    return model, preds, trues


# ---- Run training ----
train_model(model, train_loader, val_loader, test_loader, pred_len=size[2])