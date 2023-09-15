import torch
import numpy as np
from torch import nn 

device = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')

def pgd_attack(model, images, labels, eps, alpha=0.375, iters=20):
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

def fgsm_attack(model, images, labels, eps):
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