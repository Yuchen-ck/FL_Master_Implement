import torch

from torch import nn

from torch.utils.data import TensorDataset ,DataLoader
import itertools

from mnist_preprocessing import *
from args import argparse_
args = argparse_()


device = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')


def perturb_loader(perturbed_data, training_loader):
    perturbed_data_ = perturbed_data[0].cpu()
    perturbed_target_ = perturbed_data[1].cpu()
    perturb_dataset = TensorDataset(perturbed_data_, perturbed_target_)
    perturb_loader = DataLoader(perturb_dataset, batch_size = args.local_defense_bs, shuffle=True, num_workers=2) #Batch_size == 1440
    print(len(perturb_loader))
    combined_dataloader = iter(itertools.chain(perturb_loader, training_loader))
    
    return combined_dataloader


def update_weights(model, user_id ,perturbed_data):
    model.to(device)
    model.train()
    
    # Set the criterion, batch size, trainloader, etc.
    criterion = nn.CrossEntropyLoss()
    batch_size = args.local_train_bs

    if args.nonIID_mode == 'skew':
        train_loader = load_nonIID_skew(user_id)
    elif args.nonIID_mode == 'n-shards':
        train_loader = load_nonIID_n_shards(user_id)
    elif args.nonIID_mode == 'iid':
        train_loader = load_iid(user_id) 

    else:
        raise NotImplementedError
    
    
    # Set optimizer for the local updates
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr ,momentum = 0.9 ,weight_decay = 1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    total, correct, index, index_pgd = 0.0, 0.0, 0.0, 0
    epoch_loss = []

    for iter in range(args.client_epoch):
        batch_loss = []

        print("server algorithm，local不需要train髒資料阿")       
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            model.zero_grad()
            log_probs = model(images)
            
            _, pred_labels = torch.max(log_probs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
            
            loss = criterion(log_probs, labels)
            loss.backward()
            optimizer.step()

            index += -(loss.sum().item()) * len(images)
            index_pgd += len(images)

            if (batch_idx % 100 == 0):
                print(f'epoch: {iter:2}  batch: {batch_idx:4} [{32*batch_idx:6}/ 12000]  loss: {loss.item():10.8f}')

            batch_loss.append(loss.item())

        epoch_loss.append(sum(batch_loss)/len(batch_loss))
        scheduler.step()

       

    # 回傳的不是純model，是weight
    w = model.state_dict()  
    loss = sum(epoch_loss) / len(epoch_loss) 
    ide = index / len(train_loader)
    idx_train = correct/total
    pp_index = index_pgd/ len(train_loader)
    local_len = len(train_loader) * batch_size #加上defense data會更好

    print(f"用戶的資料量: {local_len}")

    return w, loss, ide, idx_train, pp_index ,local_len


def update_weights_prox(model, global_model ,user_id ,perturbed_data):
    model.to(device)
    global_model.to(device)
    model.train()
    global_model.train()
    
    # Set the criterion, batch size, trainloader, etc.
    criterion = nn.CrossEntropyLoss()
    batch_size = args.local_train_bs

    if args.nonIID_mode == 'skew':
        train_loader = load_nonIID_skew(user_id)
    elif args.nonIID_mode == 'n-shards':
        train_loader = load_nonIID_n_shards(user_id)
    elif args.nonIID_mode == 'iid':
        train_loader = load_iid(user_id) 

    else:
        raise NotImplementedError

    
    
    # Set optimizer for the local updates
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr ,momentum = 0.9 ,weight_decay = 1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


    total, correct, index, index_pgd = 0.0, 0.0, 0.0, 0
    epoch_loss = []

    # Run the training batches
    for iter in range(args.client_epoch):
        batch_loss = []
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            model.zero_grad()
            log_probs = model(images)
            
            _, pred_labels = torch.max(log_probs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
            
            # Update parameters
            optimizer.zero_grad()
            # compute proximal_term
            proximal_term = 0.0
            for w, w_t in zip(model.parameters(), global_model.parameters()):
                proximal_term += (w - w_t).norm(2)
            
            mu = 0.01
            loss = criterion(log_probs, labels) + (mu / 2) * proximal_term
            loss.backward()
            optimizer.step()

            # Print interim results
            index += -(loss.sum().item()) * len(images)
            index_pgd += len(images)

            if (batch_idx % 100 == 0):
                print(f'epoch: {iter:2}  batch: {batch_idx:4} [{32*batch_idx:6}/ 12000]  loss: {loss.item():10.8f}')

            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss)/len(batch_loss))
        scheduler.step()

    # 回傳的不是純model，是weight
    w = model.state_dict()  
    loss = sum(epoch_loss) / len(epoch_loss) 
    ide = index / len(train_loader)
    idx_train = correct/total
    pp_index = index_pgd/ len(train_loader)
    local_len = len(train_loader) * batch_size

    print(f"用戶的資料量: {local_len}")

    return w, loss, ide, idx_train, pp_index ,local_len