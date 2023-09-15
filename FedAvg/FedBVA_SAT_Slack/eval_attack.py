import torch.optim as optim
import torch
from torch import nn
from torchvision import datasets, transforms
from torchvision.transforms import Resize
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# datalaoder and 
from mnist_preprocessing import  load_test_dataloader
from mnist_model import MNIST_Net_paper

import csv

device = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')

def pgd_attack(model, images, labels, eps=0.3, alpha=0.375, iters=20):
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

def fgsm_attack(model, images, labels, eps=0.3):
    images = images.to(device)
    labels = labels.to(device)
    loss = nn.CrossEntropyLoss()

    images.requires_grad = True

    outputs = model(images)
    cost = loss(outputs, labels).to(device)

    grad = torch.autograd.grad(cost, images, retain_graph=False, create_graph=False)[0]

    adv_images = images + eps*grad.sign()
    adv_images = torch.clamp(adv_images, min=0, max=1).detach_()

    return adv_images


def test_attack(attack_function, model, dataloader, epsilon, device):
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        adv_images = attack_function(model, images, labels, eps=epsilon)
            
        outputs = model(adv_images)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print(f'Epsilon: {epsilon}, Test Accuracy: {correct / total}')
    return correct / total

def test_clean(model, dataloader):
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print(f'Test Accuracy: {correct / total}')
    return correct / total


if __name__ == '__main__':
    
    change_name = 'round_19_aggregation'
    model_path = "./SFAT_save_model/" + change_name + ".pt"

    testing_path = "./"+change_name +".csv"
    # Load MNIST test dataset
    robust_test_loader = load_test_dataloader()

    model = MNIST_Net_paper().to(device)  # Initialise your model here
     
    model.load_state_dict(torch.load(model_path))  # Load the trained model state
    model.eval()
    
    clean_acc = test_clean(model, robust_test_loader)
    print(clean_acc)

    results = []
    results.append(('clean', "-", clean_acc))
    
    # epsilons_fgsm = [0.1, 0.2, 0.3]
    # epsilons_pgd = [0.1, 0.2, 0.3]

    epsilons_fgsm = epsilons_pgd =[0.3]

    print(change_name)

    for eps in epsilons_fgsm:
        acc = test_attack(fgsm_attack, model, robust_test_loader, eps, device)
        results.append(('fgsm', eps, acc))
        
    for eps in epsilons_pgd:
        acc = test_attack(pgd_attack, model, robust_test_loader, eps, device)
        results.append(('pgd-20', eps, acc))
        
    # # CW-L2 attack with default parameters
    # acc = test_attack(cw_l2_attack, model, robust_test_loader, None, device)
    # results.append(('cw-l2', None, acc))
    

    # # Write results to CSV
    # with open(testing_path, 'a', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["Attack", "Epsilon", "Accuracy"])
    #     writer.writerows(results)