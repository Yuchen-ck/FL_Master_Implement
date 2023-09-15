# -*- coding:utf-8 -*-
"""
@Time: 2022/02/14 12:11
@Author: KI
@File: fedavg-pytorch.py
@Motto: Hungry And Humble
"""
# from uv_preprocessing_ import *
# from uv_local_train import *
# from uv_model import *
from fedAVG import *
from log_utils import *
from args import argparse_
import torch
import time
import datetime

today = datetime.datetime.today()

def write_into_txt(txt_path ,wrtie_list):
    f = open(txt_path, "a") #更改路徑!!!
    f.write(str(wrtie_list)+"\n")
    f.write("")
    f.close()
    print("Write into the file")


def training_times(start,end):
    seconds = end-start
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return (h, m, s)


def main(i):
    args = argparse_()

    start = time.time() #開始計時
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # (self, clients_num ,device,client_epoch,model,optimizer,lr,perturb_rate,attack,attack_fixed,eps)
    fed = FedAvg(clients_num = args.total_clients , model_name = args.model_name , i = i)
    
    # 執行演算法
    fed.server(r = args.communicate_round , client_rate = args.client_rate)
    
    model,test_r2_score = fed.server_test()
  
    model_name = f"./Final_Model_{i}.pt"
    torch.save(model.state_dict(), model_name)

    end = time.time()  #結束計時

    train_times = training_times(start,end)   #印出訓練總長度 

    total_time  = "- Running time: %d:%02d:%02d" % train_times
    write_into_txt('./訓練時間.txt' , total_time )

    
if __name__ == '__main__':
    for i in range(1,3):
        main(i)
    