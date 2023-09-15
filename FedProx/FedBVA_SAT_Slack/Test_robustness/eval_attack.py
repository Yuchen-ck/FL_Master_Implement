import csv
import torch
import torch.nn as nn

from mnist_model import MNIST_Net_paper
from mnist_preprocessing import *
from evaluate_attack  import *

def calculate_accuracy(eps, attack, model_path):
    print("---------------------------------------------")
    
    test_loader = load_test_dataloader()

    model = MNIST_Net_paper().cuda()

    model.load_state_dict(torch.load(model_path))

    model.eval()

    correct_predictions = 0
    total_predictions = 0
    criterion = nn.CrossEntropyLoss()

    if attack == 'none':
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.cuda(), y.cuda()
                output = model(X)
                predicted = torch.argmax(output, dim=1)
                total_predictions += y.size(0)
                correct_predictions += (predicted == y).sum().item()
    else:
        for X, y in test_loader:
            X, y = X.cuda(), y.cuda()

            if attack == 'pgd-20':
                Adv_x = pgd_attack(model,X, y ,eps)

            elif attack == 'fgsm':
                Adv_x = fgsm_attack(model,X, y ,eps)

            with torch.no_grad():
                output = model(Adv_x)
                predicted = torch.argmax(output, dim=1)
                total_predictions += y.size(0)
                correct_predictions += (predicted == y).sum().item()

    accuracy = correct_predictions / total_predictions

    print(f'Attack: {attack}, Epsilon: {eps}, Accuracy: {accuracy}')

    return accuracy


def main(model_path ,model_name):
    eps = 0
    fgsm_eps_list = [0.25 ,0.3 ,0.35]
    pgd_eps_list = [0.1 ,0.2 ,0.3]
    attack_list = ['none', 'fgsm', 'pgd-20']

    results = {'Model': model_name}

    for attack in attack_list:
        if attack == 'fgsm':
            for eps in fgsm_eps_list:
                fgsm_accuracy = calculate_accuracy(eps, attack ,model_path)
                results[f'{attack}_{eps}'] = fgsm_accuracy

        elif attack == 'none':
            normal_accuracy = calculate_accuracy(eps, attack ,model_path)
            results[f'{attack}_0'] = normal_accuracy

        elif attack == 'pgd-20':
            for eps in pgd_eps_list:
                pgd_accuracy = calculate_accuracy(eps, attack ,model_path)
                results[f'{attack}_{eps}'] = pgd_accuracy

    return results


if __name__ == "__main__":
    result_path = f'./avg_Server_slack_result.csv'
    all_results = []
    for i in range(0,3):
        # model_path = f'../SFAT_save_model/round_1_aggregation.pt'
        model_path = f'../Final_Model_{i}.pt'
        model_name = f'Final_Model_{i}'
        all_results.append(main(model_path ,model_name))

    keys = all_results[0].keys()
    with open(result_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(keys)
        for result in all_results:
            writer.writerow(result.values())


# if __name__ == "__main__":
#     for i in range(0,2):
#         model_path = f'../Final_Model_{i}.pt'
#         result_path = f'../result.csv'
#         main(model_path ,result_path)
   
