from mnist_preprocessing import *
import copy
import torch
from torch.utils.data import DataLoader ,TensorDataset
from torch import nn
from args import argparse_
args = argparse_()

# BV attack：沒有cyclic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def perturb_loader(perturbed_data):
    perturbed_data_ = perturbed_data[0].cpu()
    perturbed_target_ = perturbed_data[1].cpu()
    perturb_dataset = TensorDataset(perturbed_data_, perturbed_target_)
    perturb_loader = DataLoader(perturb_dataset, batch_size = args.server_defense_bs , shuffle=False, num_workers=0) #Batch_size == 1440
    
    return perturb_loader


def defense_adv_train(model, perturbed_data):
    print('=======Start to do Adversarial Training in Server.=======')
    model.to(device)
    model.train()
    perturb_bva_loader = perturb_loader(perturbed_data)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    total, correct, index, index_pgd = 0.0, 0.0, 0.0, 0
    epoch_loss = []

    for i in range(args.server_AT_epoch):
        batch_loss = []
        for b, (X_train, y_train) in enumerate(perturb_bva_loader):
            b+=1
            (X_train, y_train) = (X_train.to(device), y_train.to(device))

            optimizer.zero_grad()
            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()
            
            _, pred_labels = torch.max(y_pred, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, y_train)).item()
            total += len(y_train)

            index += -(loss.sum().item()) * len(X_train)
            index_pgd += len(X_train)

            print(f'epoch: {i:2}  batch: {b:4} [{b*args.server_defense_bs:6}/{len(perturb_bva_loader)*args.server_defense_bs}]  loss: {loss.item():10.8f}')

            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss)/len(batch_loss))
        scheduler.step()

    w = model.state_dict()  
    loss = sum(epoch_loss) / len(epoch_loss) 
    ide = index / len(perturb_bva_loader)
    idx_train = correct/total
    pp_index = index_pgd/ len(perturb_bva_loader)
    local_len = len(perturb_bva_loader) * args.server_defense_bs

    print(f"用戶的資料量: {local_len}")

    return w, loss, ide, idx_train, pp_index ,local_len

