
import torch
import copy
from mnist_test import test_Fashion ,test_loss_Fashion ,test_robust_Fashion ,test_pgd_Fashion

def every_round_clean_test(folder_path ,agg_model ,t):
    cr_server_model = copy.deepcopy(agg_model)
    cr_server_model.eval()
    cr_clean_acc = test_Fashion(cr_server_model)
    cr_clean_loss = test_loss_Fashion(cr_server_model)

    model_name = folder_path + f"/round_{t}_aggregation.pt"
    if cr_clean_acc != 0:
        torch.save(cr_server_model.state_dict(), model_name)
        
    return cr_clean_acc ,cr_clean_loss


def every_round_robust_test(agg_model,perturb_test_loader):
    cr_server_model = copy.deepcopy(agg_model)
    cr_server_model.eval()
    every_round_robust_acc = test_robust_Fashion(cr_server_model,perturb_test_loader)
        
    return every_round_robust_acc


def every_round_pgd_test(agg_model,test_pgd_loader):
    cr_server_model = copy.deepcopy(agg_model)
    cr_server_model.eval()
    cr_pgd_acc = test_pgd_Fashion(cr_server_model,test_pgd_loader)
        
    return cr_pgd_acc

def write_into_txt(txt_path ,wrtie_list):
    f = open(txt_path, "a") #更改路徑!!!
    f.write(str(wrtie_list)+"\n")
    f.write("")
    f.close()
    print("Write into the file")



