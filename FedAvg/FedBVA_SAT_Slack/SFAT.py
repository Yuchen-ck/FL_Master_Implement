import copy
import torch
import numpy as np
from utils import write_into_txt

# FedAvg
def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        # print(key)
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

# # FedAvg unequal
def average_weights_unequal(w, idx_num):
    """
    Returns the average of the weights.
    """

    weight_num = []
    # print(w)
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        
        w_avg[key] = w_avg[key] * float(idx_num[0]*len(w)/sum(idx_num))
        for i in range(1, len(w)):
            
            print(f"Local{i}:印出權重")
            
            print(f'local長度: {len(w)/sum(idx_num)}')
            print(f'未加權local權重: {idx_num[i]*len(w)/sum(idx_num)}')
            
            print("-------------------------------------------")
            weight_num.append(idx_num[i]*len(w)/sum(idx_num))
            w_avg[key] += w[i][key] * float(idx_num[i]*len(w)/sum(idx_num))

        print("========")

        w_avg[key] = torch.div(w_avg[key], len(w))
    print(weight_num[:6])
    return w_avg
    

# SFAT                       
def average_weights_alpha(w, lw, idx, p):
    """
    Returns the weighted average of the weights.
    """
    # 1. 創建一個與第一個模型權重相同的字典 w_avg
    w_avg = copy.deepcopy(w[0])
    

    # 2. 
    for key in w_avg.keys():
        cou = 0
        if (lw[0] >= idx):
            w_avg[key] = w_avg[key] * p
            
        for i in range(1, len(w)):
            if (lw[i] >= idx) and (('bn' not in key)):
                w_avg[key] = w_avg[key] + w[i][key] * p
            else:
                cou += 1 
                w_avg[key] = w_avg[key] + w[i][key]
        w_avg[key] = torch.div(w_avg[key], cou +(len(w)-cou) * p)
    return w_avg


# # SFAT unequal
def average_weights_alpha_unequal(w, max_l, idx, p, idx_num):
    """
    Returns the weighted average of the weights.
    """
    weight_num = []
    client_add_list = list()
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        cou = 0

        #決定起始點
        if (lw[0] >= idx):
            calculation_plus = p * float(idx_num[0]*len(w)/sum(idx_num))
            w_avg[key] = w_avg[key] * calculation_plus
            
        else:
            print("比較小")
            calculation_plus = float(idx_num[0]*len(w)/sum(idx_num))

            w_avg[key] = w_avg[key] *calculation_plus

        for i in range(1, len(w)): #len(w) = 7

            if i == 5:
                calculation_plus = p * float(idx_num[0]*len(w)/sum(idx_num))
                w_avg[key] = w_avg[key] + w[i][key] * p * float(idx_num[i]*len(w)/sum(idx_num))
                weight_num.append(calculation_plus)
                client_add_list.append(f"local_{i} 有加權")
                 
            else:
                cou += 1 
                calculation_plus = p * float(idx_num[0]*len(w)/sum(idx_num))
                w_avg[key] = w_avg[key] + w[i][key] * float(idx_num[i]*len(w)/sum(idx_num))
                weight_num.append(idx_num[i]*len(w)/sum(idx_num))
                
        print("========")
        w_avg[key] = torch.div(w_avg[key], cou+(len(w)-cou)*p)
    
    print(weight_num[:6])

    write_into_txt("./權重分配.txt" ,weight_num[:6])

    # print(client_add_list)
    
    return w_avg  





