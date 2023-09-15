import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mnist_preprocessing import load_test_dataloader
from args import argparse_
args = argparse_()


device = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')

# 1. clean acc
def test_Fashion(model):
    model.to(device)
    model.eval()
    # print("測試準確度")
    correct = 0
    test_loader = load_test_dataloader()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            prediction = output.data.max(1)[1]
            correct += prediction.eq(target.data).sum()

        # print(len(test_loader.dataset))

    # print('\nTesting without attack - Accuracy: {:.2f}%'.format(100. * correct / len(test_loader.dataset)))
    test_acc = 100. * correct / len(test_loader.dataset)

    return test_acc.item() # tensor(87.1000, device='cuda:0') to float

# 2. clean loss
def test_loss_Fashion(model):
    model.to(device)
    model.eval()
    test_loss = 0
    test_loader = load_test_dataloader()
    criterion = torch.nn.CrossEntropyLoss()  # define loss function

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # sum up batch loss
            loss = criterion(output, target)
            test_loss += loss.item()  # get the raw value with .item()

    # calculate average loss over all samples
    test_loss /= len(test_loader.dataset)
    
    print('\nTest set: Average loss: {:.4f}'.format(test_loss))

    return test_loss



def pgd_attack(model, images, labels, eps=0.3, alpha=0.01, iters=20):
    images = images.to(device)
    labels = labels.to(device)
    loss = nn.CrossEntropyLoss()

    ori_images = images.data

    for i in range(iters) :  
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, labels).to(device)
        cost.backward()

        adv_images = images + alpha*images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()

    return images

# 3. BVA acc
def test_robust_Fashion(model ,test_loader):
    model.to(device)
    model.eval()
    # print("測試準確度")
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            prediction = output.data.max(1)[1]
            correct += prediction.eq(target.data).sum()

        # print(len(test_loader.dataset))

    # print('\nTesting without attack - Accuracy: {:.2f}%'.format(100. * correct / len(test_loader.dataset)))
    test_acc = 100. * correct / len(test_loader.dataset)

    return test_acc.item() # tensor(87.1000, device='cuda:0') to float 

# 4. PGD acc
def test_pgd_Fashion(model, test_loader, eps=0.3, alpha=0.01, iters=20):
    model.to(device)
    model.eval()
    correct = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        # Perform the PGD attack
        attacked_data = pgd_attack(model, data, target, eps, alpha, iters)

        # Make a prediction using the attacked data
        with torch.no_grad():  # Now gradients computation is off only when making predictions
            output = model(attacked_data)
            prediction = output.data.max(1)[1]
            correct += prediction.eq(target.data).sum()

    test_acc = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Accuracy under PGD attack: {:.2f}%'.format(test_acc))

    return test_acc.item()

# for final testing
def perturb_test_loader(perturbed_data):
    perturbed_data_ = perturbed_data[0].cpu()
    perturbed_target_ = perturbed_data[1].cpu()
    perturb_dataset = TensorDataset(perturbed_data_, perturbed_target_)
    perturb_test_loader = DataLoader(perturb_dataset, batch_size = args.local_test_bs, shuffle=True, num_workers=2) #Batch_size == 1440
    
    return perturb_test_loader




