import argparse

def argparse_():

    # Adjust the parameters right here.

    # 1. FL_setting
    dataset = "MNIST"
    nonIID_mode = 'n-shards' 
    communication_round = 2 #15
    total_clients = 2  #5
    client_rate = 1 #1
    client_epoch = 2  #5

    # 2. Local_Train setting
    local_train_bs = 256
    local_defense_bs = server_defense_bs = 256 # train_bs和defense_bs一致，因為要加起來一起訓練
    local_test_bs = 16
    model_name = 'MNIST_Net_paper'
    optimizer ='Adam'
    lr = 0.01

    # 3. Local_Train Algo setting4
    FedProx = 1  # 0:FedAvg

    # 4. Server Aggregation setting
    aggregate_algorithm = 'SFAT_unequal_loss'  #average_unequal #SFAT_unequal #SFAT_unequal_loss

    # 5. Test Robustness setting
    test_attack = 'none'
    
    parser = argparse.ArgumentParser(description='Adversarial Robusteness in Federated Learning Learning')
    # ============Save_Result setting========
    FAT_save_model = "./FAT_save_model"
    SFAT_save_model = "./SFAT_save_model"

    parser.add_argument('--FAT_folder_path',default = FAT_save_model,  help='FAT folder path')
    parser.add_argument('--SFAT_folder_path',default = SFAT_save_model, help='SFAT folder path')

    clean_result  = "./clean_result.txt"
    clean_loss_result = './clean_loss.txt'
    robust_result  = "./robust_result.txt"
    pgd_result = "./pgd_result.txt"

    parser.add_argument('--clean_result_txt',default = clean_result,  help='clean_result txt')
    parser.add_argument('--clean_loss_result_txt',default = clean_loss_result, help='clean_loss_result txt')
    parser.add_argument('--robust_result_txt',default = robust_result, help='pgd_result txt')
    parser.add_argument('--pgd_result_txt',default = pgd_result, help='clean_loss_result txt')
    
    # print img round
    print_img_round = 1
    parser.add_argument('--print_img_round', type=int, default = print_img_round, help='')

    # ============Non-IID setting ============

    parser.add_argument('--nonIID_mode', default = nonIID_mode, choices=['skew', 'n_shards' ,'iid'], help='model architecture')


    # ============FL_Train setting============
    parser.add_argument('--total_clients', default = total_clients, type=int, help='total client numbers')
    parser.add_argument('--communicate_round', default = communication_round, type=int, help='communication Round')
    parser.add_argument('--client_epoch', default = client_epoch, type=int, help='client training epoch')
    parser.add_argument('--local_train_bs', type=int, default = local_train_bs ,help="local batch size: B, 128 for SVHN and CIFAR-100")
    parser.add_argument('--local_test_bs', type=int, default = local_test_bs ,help="local batch size: B, 128 for SVHN and CIFAR-100")
    parser.add_argument('--local_defense_bs', type=int, default = local_defense_bs ,help="local batch size: B, 128 for SVHN and CIFAR-100")
    parser.add_argument('--server_defense_bs', type=int, default = server_defense_bs ,help="local batch size: B, 128 for SVHN and CIFAR-100")

    parser.add_argument('--client_rate', default = client_rate, type=float,help='client joint aggregation rate')
    parser.add_argument('--model_name', default = model_name, choices=['CNN', 'ResNet18' ], help='model architecture')
    parser.add_argument( '--optimizer', default = optimizer, choices=['Adam', 'RMSprop'], help='optimizer select')
    parser.add_argument( '--lr', default = lr, type=float, help='client learning rate')

    # ============FedProx setting============
    parser.add_argument('--FedProx', default = FedProx, type=int, help='client learning rate')
    
    # ============attack (Test Robustness) setting==============
    parser.add_argument('--attack', default = test_attack, choices=['none','fgsm', 'pgd','others'], help='adversarial attack type')
    
    # ============aggregate setting==============
    parser.add_argument( '--aggregate_algorithm', default = aggregate_algorithm, choices=['average','SFAT'], help='Aggregate Algorithm')
    
    # ***SFAT***
    parser.add_argument('--topk',type=int, default=1, help='top client to be upweight')
    parser.add_argument('--pri', type=float, default=1.4, help='weight for (1+alpha)/(1-alpha): 1.2, 1.4, 1.6 ...')
    
    # ============ BV_attack  setting==============
    BV_FGSM_alpha = 0.3
    parser.add_argument('--BV_FGSM_alpha', type=float, default= BV_FGSM_alpha, help='BV_FGSM_alpha')
    
    parser.add_argument('--c2', type=float, default=0.01, help=' c2: balance factor  - 0.01')
    
    # 論文固定的eps
    mnist_eps = 0.3
    parser.add_argument('--mnist_eps', type=float, default= 0.3, help=' c2: balance factor  - 0.01')
    parser.add_argument('--alpha', type=float, default= 0.375, help=' c2: balance factor  - 0.01')

    defense_data_numbers = 2 #300 #(3000/60000 = 0.05) (5%資料放進server)
    test_numbers = 16
    # test_numbers = int(defnse_data_numbers * 0.2)
    # server data numbers
    parser.add_argument('--defnse_data_numbers',type=int, default = defense_data_numbers, help='top client to be upweight')
    parser.add_argument('--test_numbers',type=int, default = test_numbers, help='top client to be upweight')

    # ============ Draw images  setting==============
    robust_img_folder = "./bva_img/"
    pgd_img_folder = "./pgd_img/"

    parser.add_argument('--robust_img_folder',type=str, default = robust_img_folder, help='top client to be upweight')
    parser.add_argument('--pgd_img_folder',type=str, default = pgd_img_folder, help='top client to be upweight')

    # ===============Server Training ============
    client_at_alpha = 0.2  #[0.2 * 5 = 1] [0.4 * 5 = 2] 
    server_AT_epoch = 50
    parser.add_argument('--client_at_alpha', type=float, default= client_at_alpha, help=' Default : client_at_alpha = 0.4')
    parser.add_argument('--server_AT_epoch',type=int, default = server_AT_epoch, help='top client to be upweight')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    # from args import argparse_
    args = argparse_()
    print(args.client_at_alpha)

    # pass