import torch
import torch.nn.functional as F

from BVA_attack  import *
from args import argparse_

args = argparse_()
# data_grads = data_grads_bias + c2 * data_grads_variance


def fgsm_attack(data ,data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = data + args.alpha * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, -args.mnist_eps, args.mnist_eps)
    return perturbed_image.detach()



# 再想看看能不能放PGD進去勒
'''
def pgd_attack(model, images, labels, epsilon, alpha, iters):
    # Clone the original images to not affect them
    perturbed_image = images.clone().detach()
    
    for i in range(iters):
        # Set requires_grad attribute of tensor to make sure we get gradients
        perturbed_image.requires_grad = True
        
        # Forward pass
        outputs = model(perturbed_image)
        model.zero_grad()
        
        # Calculate loss
        loss = F.cross_entropy(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Calculate the sign of gradient
        data_grad = perturbed_image.grad.data
        sign_data_grad = data_grad.sign()
        
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = perturbed_image.detach() + alpha*sign_data_grad

        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        
        # Return the perturbed image into the epsilon ball
        perturbation = perturbed_image - images
        perturbation = torch.clamp(perturbation, -epsilon, epsilon)
        perturbed_image = images + perturbation

        # Again clamping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)

    return perturbed_image.detach()
'''

if __name__ == "__main__":
    # load_model_path = './2023-02-06_0.96486.pt'
    # model = small_ANN().cuda()
    # model.load_state_dict(torch.load(load_model_path))
    # model.eval()
    pass
    
    
    

