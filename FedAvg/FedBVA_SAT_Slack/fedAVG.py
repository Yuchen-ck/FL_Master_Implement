import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import copy
import random
import torch 
import numpy as np

from mnist_train import *
from mnist_model import MNIST_Net_paper
from mnist_test import *
from SFAT import *
from SFAT_change import *
from BVA_attack import get_BVD_adv_examples ,get_BVD_test_data
from utils import *
from server_AT import defense_adv_train
from print_img import * 

from args import argparse_
args = argparse_()

# if 'SFAT' in args.aggregate_algorithm:
#     string_algorithm = 'SFAT'
# elif 'average' in args.aggregate_algorithm:
#     string_algorithm = 'FAT'
# else:
#     raise NotImplementedError

string_algorithm = " "

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FedAvg:
    def __init__(self, clients_num , model_name ,i): # def __init__(self, args):
        

        # 0.0 創建每次執行的資料夾，並把相關數據存進去
        self.folder_path = "./" + string_algorithm + f"_save_model_{i}" 
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        # 0.1 保存每回合的模型
        self.folder_model = "./" + string_algorithm + f"_save_model_{i}" +"/model" 
        if not os.path.exists(self.folder_model):
            os.makedirs(self.folder_model)
        
        # 0.1 保存每回合的結果
        self.folder_result = "./" + string_algorithm + f"_save_model_{i}" +"/result" 
        if not os.path.exists(self.folder_result):
            os.makedirs(self.folder_result)

        self.client_number = clients_num

        # 1. client_model和global_model的初始設定
        self.global_model = MNIST_Net_paper()
        self.global_weights = self.global_model.state_dict()

        server_defense_train_module = int(args.client_at_alpha * args.total_clients)
        
        self.client_model = []
        self.client_model = [copy.deepcopy(self.global_model) for i in range(args.total_clients + server_defense_train_module)]
        # print(self.client_model[0])

        self.perturbed_data = None
        self.test_perturbed_data = None
        
        self.clients_list = []  #5 clients
        for i in range(self.client_number):
            self.clients_list.append("client_"+str(i))
        
        # 指定模型形式
        if model_name == "MNIST_Net_paper":
            self.global_model = MNIST_Net_paper() 
            # self.nn = MNIST_Net_paper() 
        else:
            raise NotImplementedError
        

        self.server_at_model = MNIST_Net_paper()

          
    def server(self, r, client_rate):
        self.r = r  #the number of communication rounds
        self.client_rate = client_rate

        self.clean_acc_list = list()
        self.clean_loss_list = list()
        self.robust_acc_list = list()

        # communication round從這裡開始啦!
        for t in range(self.r):

            print('round', t + 1, ':') #最外圈for loop
            # sampling
            m = np.max([int(self.client_rate * self.client_number), 1])
            index = random.sample(range(0, self.client_number), m)  #St #random set of m clients
            
            self.idt = []
            self.local_weights  = []
            self.local_losses = []
            self.idx_num = []
            self.ctr = 0
            self.local_len_list = []

            # 1. dispatch
            self.dispatch(index)

            # 2. local updating
            self.client_update(index ,self.perturbed_data)

            # 3. perturbate_data (參考:Fed_BVA https://github.com/jwu4sml/FedBVA)

            # if t == 0:
            self.perturbed_data = self.perturbate_data(t)

            # 3.1  For testing in every rc.(生成每一輪的對抗樣本測試)
            self.test_perturbed_data = self.perturbate_test_data()

            # 4. defense adversarial training
            if self.perturbed_data is not None and t > 5 :
                self.server_defense_train(self.perturbed_data)
            else:
                print("直接聚合")


            # 5. aggregation
            self.global_model = self.aggregation(index,t)
            
            

        return self.global_model

    #1.
    def dispatch(self, index):
        print(index) 
        #print(self.nns[0].len) #從這裡改

        # 将一组神经网络的参数（self.nn.parameters()）复制到另一组神经网络（self.nns[j].parameters()）。
        for j in index:
            for old_params, new_params in zip(self.client_model[j].parameters(), self.global_model.parameters()):
                old_params.data = new_params.data.clone()
                #print(old_params.data)
        print("過")
        # 這裡應該要大改: 0518!!!

    #2. 
    def client_update(self, index ,perturbed_data):  # update Client Model
        # total_clients : args.total_clients

        #訓練local模型
        print("MNIST手寫資料集/分類問題")
        for k in index:
            print("The client_{} start to train(clean)".format(k))
            if args.FedProx == 0 :  #不是FedProx                                             #copy model is better.
                w ,loss, ide, idx_train, pp_index ,user_data_len = update_weights(model = copy.deepcopy(self.client_model[k]) ,user_id = k ,perturbed_data = perturbed_data) 
            elif args.FedProx == 1:
                w ,loss, ide, idx_train, pp_index ,user_data_len = update_weights_prox(model = copy.deepcopy(self.client_model[k]) ,global_model = copy.deepcopy(self.global_model) ,user_id = k ,perturbed_data = perturbed_data)
            else:
                raise NotImplementedError
            
            self.local_weights.append(copy.deepcopy(w))
            self.local_losses.append(copy.deepcopy(loss))
            
            self.idx_num.append(user_data_len)

            if "_unequal" not in args.aggregate_algorithm:
                self.idt.append(loss)
            else: 
                # 資料不平均
                self.idt.append(loss*(self.idx_num[self.ctr]))

            self.ctr = self.ctr +1

    #3. perturbate_data (參考:Fed_BVA https://github.com/jwu4sml/FedBVA)
    def perturbate_data(self , t ):
        import time
        start_time = time.time()  # Start time

        print("---------Start to generate the BV_attack for defensing.-----------")
        self.perturbed_data =  get_BVD_adv_examples(self.local_weights) 
        print("----------------------THE END-----------------------") 
        
        end_time = time.time()
        elapsed_time = end_time - start_time  # Elapsed time

        minutes, seconds = divmod(elapsed_time, 60)  # Convert time to minutes and seconds
        str_time = f"Cr {t} generating times: {int(minutes)} minutes and {int(seconds)} seconds."
        
        write_into_txt(self.folder_path +'/generate_perturbation.txt' ,str_time)

        return self.perturbed_data


    def perturbate_test_data(self):
        print("---------Start to generate the BV_attack for testing.-----------")
        self.test_perturbed_data  =  get_BVD_test_data(self.local_weights) # seed=42:defense_loader的隨機種子
        print("----------------------THE END-----------------------") 
        return self.test_perturbed_data 
    

    # 4.Sever Adversarial Training
    def server_defense_train(self ,perturbed_data):
        print("中心伺服器進行對抗訓練")
        # 對抗式防禦
        # args.client_at_alpha = 0.4
        server_at_module_number = int(args.total_clients * args.client_at_alpha)

        print(args.total_clients)
        print(args.total_clients+server_at_module_number)
        
        for i in range(args.total_clients ,args.total_clients+server_at_module_number):
            w ,loss, ide, idx_train, pp_index ,user_data_len = defense_adv_train(copy.deepcopy(self.server_at_model) ,perturbed_data)
            print(loss)
            # 把加躁後的權重納入聚合
            self.local_weights.append(copy.deepcopy(w))
            self.local_losses.append(copy.deepcopy(loss))
            
            self.idx_num.append(user_data_len)

            if "_unequal" not in args.aggregate_algorithm:
                self.idt.append(loss)
            else: 
                # 資料不平均
                self.idt.append(loss*(self.idx_num[self.ctr]))

            self.ctr = self.ctr +1
        
        print("idt_list:")
        print(self.idt)
        print(len(self.idt))
        
    #4. 聚合算法在這裡!
    def aggregation(self, index ,t): #question
        
        #print local loss
        print(self.local_losses)
        local_loss_dict = {index: value for index, value in enumerate(self.local_losses)}
        write_into_txt(self.folder_path + "./loss對照表.txt",str(local_loss_dict))
        
        '''
        # if args.aggregate_algorithm == 'old':
        #     # 已經廢除了
        #     s = 0
        #     for j in index:
        #         # normal
        #         s += self.local_len_list[j] #s: 總資料量
        #         print(f"local_{j}:總資料量(dataloader)_{self.local_len_list[j]}")

        #     params = {}
        #     for name,  param in self.client_model[0].named_parameters(): # model的名字與参数迭代器
        #         params[name] = torch.zeros_like(param.data)   

        #     for j in index:
        #         for name, param in self.client_model[j].named_parameters(): 
        #             params[name] += param.data * (self.local_len_list[j]/ s)

        #     # self.global_model.load_state_dict(self.global_weights)
        #     for k, v in self.global_model.named_parameters():
        #         v.data = params[k].data.clone()

        #     # server_model = self.nns
        #     # print(self.nns)
        #     # print(len(self.nns))
        
        # elif args.aggregate_algorithm == 'SFAT':
            
        #     joint_client_number = len(index)

        #     idt_sorted = np.sort(self.idt)
        #     idtxnum = float('inf')
        #     idtx = args.topk
        
        #     if idtx > joint_client_number:
        #         idtx = joint_client_number
            
        #     if idtx != 0:
        #         idtxnum = idt_sorted[joint_client_number-idtx]
            
        #     if t+1 >0:
        #         print("第二個回傳值")
        #         print(self.idt)
        #         print("第三個回傳值")
        #         print(idtxnum) 
                   
        #         global_weights = average_weights_alpha(self.local_weights, self.idt, idtxnum, args.pri)
        #         #global_weights = average_weights_alpha_unequal(local_weights, idt, idtxnum, args.pri, idx_num)                
                
        #     else:
        #         global_weights = average_weights(self.local_weights)
        #         #global_weights = average_weights_unequal(local_weights, idx_num)

        # elif args.aggregate_algorithm == 'SFAT_unequal':
            
        #     joint_client_number = len(self.idt)

        #     idt_sorted = np.sort(self.idt)
        #     idtxnum = float('inf')
        #     idtx = 1

        #     if idtx >joint_client_number:
        #         idtx = joint_client_number
            
        #     if idtx != 0:
        #         idtxnum = idt_sorted[joint_client_number-idtx]

        #     if t+1 >0:
        #         # print("第二個回傳值")
        #         # print(self.idt)
        #         # print("第三個回傳值")
        #         print(idtxnum) 
                   
        #         global_weights = average_weights_alpha_unequal(self.local_weights, self.idt, idtxnum, args.pri, self.idx_num)                
                
        #     else:
        #         # global_weights = average_weights(self.local_weights)
        #         global_weights = average_weights_unequal(self.local_weights, self.idx_num)
        '''

        #取最大的loss進行加權
        if args.aggregate_algorithm == 'SFAT_unequal_loss':
            
            max_key = max(local_loss_dict.keys())

            if (t+1) % 2 == 1:
                print("加權平均")
                global_weights = average_weights_loss_alpha_unequal(self.local_weights, max_key ,args.pri, self.idx_num)                
                
            else:
                print("不加權平均")
                print(len(self.local_weights.pop()))
                global_weights = average_weights_unequal(self.local_weights, self.idx_num)


        elif args.aggregate_algorithm == 'average':
            global_weights = average_weights(self.local_weights)
        
        elif args.aggregate_algorithm == 'average_unequal':
            print(self.idx_num) #local端的資料量
            global_weights = average_weights_unequal(self.local_weights, self.idx_num)

        else:
            raise NotImplementedError

        self.global_model.load_state_dict(global_weights)
        
        # ================= Every CR testing ==============
        # 每次聚合後都要進行測試
        if self.test_perturbed_data is not None:

            # 1. clean testing
            clean_acc ,clean_loss = every_round_clean_test(self.folder_model ,self.global_model ,t) #回傳測試的r2_score
            self.clean_acc_list.append(round(clean_acc ,5))
            self.clean_loss_list.append(round(clean_loss ,5))

            write_into_txt(self.folder_result + "/"+ args.clean_result_txt ,self.clean_acc_list)
            write_into_txt(self.folder_result + "/"+ args.clean_loss_result_txt ,self.clean_loss_list)

            
            # 2. robust testing
            robust_test_loader = perturb_test_loader(self.test_perturbed_data)
            
            # Drawing BVA images in every cr.
            robust_img_path = args.robust_img_folder # ./bva_img/
            if t % args.print_img_round == 0:
                draw(robust_test_loader , t  ,robust_img_path)
            
            robust_acc = every_round_robust_test(self.global_model,robust_test_loader)
            
            self.robust_acc_list.append(round(robust_acc ,5))
            write_into_txt(self.folder_result + "/"+ args.robust_result_txt ,self.robust_acc_list)
            
            
            # # 3. PGD testing
            # test_pgd_loader = load_test_defense_dataloader(seed = 69)
            # pgd_acc = every_round_pgd_test(self.global_model ,test_pgd_loader)

            # # Drawing PGD images in every cr.
            # pgd_img_path = args.robust_img_folder  #"./pgd_img/"
            # # if t % args.print_img_round == 0:
            # # draw_pgd(test_pgd_loader , t  , pgd_img_path)
            
            # self.pgd_acc_list.append(round(pgd_acc ,5))
            # write_into_txt("./cr_result/"+ args.pgd_result_txt ,self.pgd_acc_list)


        else:
            raise NotImplementedError

        # ======================================================================

        return self.global_model



    def server_test(self):
        model = self.global_model
        server_model = copy.deepcopy(model)
        server_model.eval()
        server_name = ["server"]
        
        for server in server_name:
            server_model.name = server
            print("server最終測試結果為: ")
            test_acc = test_Fashion(model)
            print(f'{round(test_acc,3)}%')

        return model, test_acc
        

    # #測試個別client的數據
    # def global_test(self,clients_num):
    #     model = self.nn
    #     model.eval()
    #     clients_wind = ['Task1_W_Zone' + str(i) for i in range(1, clients_num)]
    #     c = clients_wind
    #     for client in c:
    #         model.name = client
    #         test_Fashion(model)

    
    # #保存個別client的模型
    # def client_model_save(self):
    #     client_model_list = []
    #     client_model_list = [copy.deepcopy(self.global_model) for i in range(args.total_clients)]
        
    #     for i in range(args.total_clients * args.client_rate):
    #         client_model_list[i].load_state_dict(self.local_weights[i])
            

