import torch.nn.functional as F
import torch
from fgsm_attack import fgsm_attack
import copy

from mnist_model import MNIST_Net_paper
from mnist_preprocessing import load_defense_dataloader ,load_test_defense_dataloader

from args import argparse_
args = argparse_()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_BIAS(client_models_weights, data, target):
    data_grads, output_mean = [], []

    client_models_list_V2 = list()

    global_model = MNIST_Net_paper().to(device)   
    client_model_list = []
    client_model_list = [copy.deepcopy(global_model) for i in range(args.total_clients)]

    for i in range(args.total_clients * args.client_rate):
        client_model_list[i].load_state_dict(client_models_weights[i])
        client_models_list_V2.append(copy.deepcopy(client_model_list[i]))
        
    for im, model in enumerate(client_models_list_V2):
        model_copy = copy.deepcopy(model)
        data_clone, target_clone = data.clone(), target.clone()
        data_clone.requires_grad = True
        output = model_copy(data_clone)
        output_mean.append(F.softmax(output, dim=1))

        loss = F.cross_entropy(output, target_clone)
        model_copy.zero_grad()
        loss.backward()
        data_grad = data_clone.grad.data
        data_grads.append(data_grad)
    output_mean = torch.mean(torch.stack(output_mean, 0), 0)
    data_grads_bias = torch.mean(torch.stack(data_grads, 0), 0)
    return data_grads_bias, output_mean.cpu().detach().numpy()


def get_VARIANCE(client_models_weights, data, output_mean):
    batch_size = data.shape[0]
 
    
    client_models_list_V2 = list()

    global_model = MNIST_Net_paper().to(device)   
    client_model_list = []
    client_model_list = [copy.deepcopy(global_model) for i in range(args.total_clients)]

    for i in range(args.total_clients * args.client_rate):
        client_model_list[i].load_state_dict(client_models_weights[i])
        client_models_list_V2.append(copy.deepcopy(client_model_list[i]))

    data_grads_variance = list()
    for im, model in enumerate(client_models_list_V2):
        data_clone = data.clone()
        data_clone.requires_grad = True
        output = F.softmax(model(data_clone), dim=1)

        data_grads_variance_subset = 0
        data_grads_previous = 0
        for ic in range(output.shape[1]):  # output.shape = [1, 10] for MNIST
            variance_multiplier = torch.log(output_mean[:, ic]) + 1
            for ib in range(batch_size):
                model.zero_grad()
                output[ib, ic].backward(retain_graph=True)
                data_grad = data_clone.grad.data - data_grads_previous
                data_grads_previous = copy.deepcopy(data_clone.grad.data)                                            #data.shape[2]: 28
                data_grads_variance_subset += data_grad * variance_multiplier.view(batch_size, 1, 1, 1).repeat(1, 1, data.shape[2], data.shape[2])
        data_grads_variance.append(data_grads_variance_subset)

    data_grads_variance = -1 * torch.mean(torch.stack(data_grads_variance, 0), 0)
    
    return data_grads_variance


def get_BVD_adv_examples(client_models ,seed = 42):
    
    defense_loader = load_defense_dataloader(seed)

    # args.c2: balance factor [default: 0.01]
    
    perturbed_data, perturbed_target = None, None
    for  (data, target) in defense_loader:
    
        data, target = data.to(device), target.to(device)
        image_range = [data.min(), data.max()]

        data_grads_bias, output_mean = get_BIAS(client_models, data, target)
        output_mean = torch.from_numpy(output_mean).float().to(device)
        data_grads_variance = get_VARIANCE(client_models, data, output_mean)
        data_grads = data_grads_bias + args.c2 * data_grads_variance
        
        perturbed_data = fgsm_attack(data, data_grads)
        perturbed_target = target

    return  perturbed_data , perturbed_target 



def get_BVD_test_data(client_models ,seed = 69):
    
    defense_loader = load_test_defense_dataloader(seed)

    # args.c2: balance factor [default: 0.01]
    perturbed_data, perturbed_target = None, None
    for  (data, target) in defense_loader:
    
        data, target = data.to(device), target.to(device)
        image_range = [data.min(), data.max()]

        data_grads_bias, output_mean = get_BIAS(client_models, data, target)
        output_mean = torch.from_numpy(output_mean).float().to(device)
        data_grads_variance = get_VARIANCE(client_models, data, output_mean)
        data_grads_variance = 0
        data_grads = data_grads_bias + args.c2 * data_grads_variance
        
        perturbed_data = fgsm_attack(data, data_grads)
        perturbed_target = target

    return  perturbed_data , perturbed_target 

if __name__ == '__main__':
    
    defense_loader = load_defense_dataloader() 
    for X , y in defense_loader:
        print(X.shape)
