# 以loss為主
# # SFAT unequal

import copy
import torch
import numpy as np
from utils import write_into_txt

def average_weights_loss_alpha_unequal(w, max_key_loss, p, idx_num):
    """
    Returns the weighted average of the weights.
    """
    weight_num = []
    client_add_list = list()
    w_avg = copy.deepcopy(w[0])
    

    p = 1.2
    p_plus = 1

    for key in w_avg.keys():
        cou = 0

      

        for i in range(0, len(w)): #len(w) = 7

            if i == max_key_loss:
                # p = 2 # p = 1.4
                calculation_plus = p * float(idx_num[0]*len(w)/sum(idx_num))
                w_avg[key] = w_avg[key] + w[i][key] * p * float(idx_num[i]*len(w)/sum(idx_num))
                weight_num.append(calculation_plus)
                client_add_list.append(f"local_{i} 有加權")
                 
            else:
                cou += 1 
                calculation_plus = p * float(idx_num[0]*len(w)/sum(idx_num))
                w_avg[key] = w_avg[key] + w[i][key] *p_plus* float(idx_num[i]*len(w)/sum(idx_num))
                weight_num.append(idx_num[i]*len(w)/sum(idx_num))
                
        w_avg[key] = torch.div(w_avg[key], (cou * p_plus) + (len(w)-cou) * p )
    
    return w_avg  