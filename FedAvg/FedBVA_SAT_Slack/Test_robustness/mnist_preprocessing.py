import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split , Subset
import numpy as np
from args import argparse_
args = argparse_()

from torch.utils.data import TensorDataset
# print(args.local_bs)

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
def load_train_dataset():
    transform = transforms.Compose([transforms.ToTensor() ,transforms.Normalize((0.5,), (0.5,))])
    
    # 使用内置函数下载mnist数据集
    train_dataset = datasets.MNIST(root='./MNIST_data', train=True, transform=transform, download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size = 128, shuffle=True)

    return train_loader

# 0. IID
def load_iid(user_id):

    # data preprocessing
    transform = transforms.Compose([transforms.ToTensor() ,transforms.Normalize((0.5,), (0.5,))])
    
    # 使用内置函数下载mnist数据集
    train_dataset = datasets.MNIST(root='./MNIST_data', train=True, transform=transform, download=True)
    
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.total_clients, rank=user_id)
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=sampler,batch_size = args.local_train_bs, shuffle=sampler is None)

    return train_loader


# 1. non-IID skew 參考:https://github.com/Xtra-Computing/NIID-Bench
def load_nonIID_skew(user_id ,num_users=5, skew=2):

    """
    從 MNIST 數據集中，非獨立同分布 (non-I.I.D) 的方式採樣客戶數據
    :param dataset: CIFAR10 數據集
    :param num_users: 用戶的數量
    :param skew: 數據分佈的偏度
    :return: 返回一個字典，每個用戶對應其相應的數據索引
    """

    transform = transforms.Compose([transforms.ToTensor() ,transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./MNIST_data', train=True, transform=transform, download=True)

    number_each_class = 60000  # MNIST 有 60000 個例子
    num_shards, num_imgs = int(num_users * 2), int(30000 / num_users)  # 計算每個用戶將擁有的數據片段數量和圖片數量
    idx_shard = [i for i in range(num_shards)]  

    # divide the data by the class labels equally among K client
    new_datas = [[] for _ in range(10)] 
    M_k = [[] for _ in range(num_users)] 

    dict_users = {i: np.array([]) for i in range(num_users)} 
    idxs = np.arange(num_shards * num_imgs)  
    # labels = dataset.train_labels.numpy()
    labels = np.array(train_dataset.targets)  

    # sort labels
    idxs_labels = np.vstack((idxs, labels)) 
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()] 
    idxs = idxs_labels[0, :] 
    
    # 將資料按照標籤均等地分配到每個用戶
    for i in range(10):
        new_datas[i] = idxs[i * number_each_class:(i + 1) * number_each_class]
         # print(len(new_datas[i]))

    for i in range(num_users):
        M_k[i] = idxs[i * int(len(idxs) / num_users):(i + 1) * int(len(idxs) / num_users)]
    
    # 進行非IID的資料分配
    kk = len(M_k[0])
    for i in range(num_users):
        for j in range(num_users):
            if i == j:
                # 如果用戶 i 和 j 相同，則該用戶從他的 M_k 片段中獲得大部分的數據
                rand_set = set(np.random.choice(M_k[j], int(kk * (100 - (num_users - 1) * skew) / 100), replace=False))
                M_k[j] = list(set(M_k[j]) - rand_set)
                for rand in rand_set:
                    dict_users[i] = np.concatenate((dict_users[i], [rand]), axis=0)
            else:
                # 如果用戶 i 和 j 不同，則該用戶只從其他用戶的 M_k 片段中獲得少量的數據
                rand_set = set(np.random.choice(M_k[j], int(kk * (skew) / 100), replace=False))
                M_k[j] = list(set(M_k[j]) - rand_set)
                for rand in rand_set:
                    dict_users[i] = np.concatenate((dict_users[i], [rand]), axis=0)

    # 印出每個用戶的數據量以及類別的數量
    labels = np.array(train_dataset.targets)
    for i in range(num_users):
        user_data_indices = dict_users[i]
        user_labels = labels[user_data_indices.astype(int)]
        unique_labels, counts = np.unique(user_labels, return_counts=True)
        
        if i == user_id:
            print(f"用戶 {i} 的數據量：", len(dict_users[i]))
            print(f"用戶 {i} 的類別數量：", dict(zip(unique_labels, counts)))
            user_data = [(train_dataset[i][0], train_dataset[i][1]) for i in user_data_indices.astype(int)]
            id_dataset = CustomDataset(user_data)

            train_loader = DataLoader(id_dataset, batch_size = args.local_train_bs, shuffle=True, num_workers=2)

            return train_loader  # 只返回指定 user_id 的 DataLoader


# 2. non-IID n-shards
def load_nonIID_n_shards(user_id):
    transform = transforms.Compose([transforms.ToTensor() ,transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./MNIST_data', train=True, transform=transform, download=True)

    # 建立一个空的索引列表，用于储存对应的样本索引
    indices = []

    # 为每个user_id定义对应的类别
    classes_dict = {
        0: [0, 1, 2, 3],
        1: [2, 3, 4, 5],
        2: [4, 5, 6, 7],
        3: [6, 7, 8, 9],
        4: [8, 9, 0, 1]
    }

    classes_for_user = classes_dict[user_id]

    for i in range(len(train_dataset)):
        # 如果当前样本的类别在该用户的类别列表中，则添加其索引
        if train_dataset.targets[i] in classes_for_user:
            indices.append(i)

    # 使用Subset从MNIST数据集中提取出特定的样本
    id_dataset = Subset(train_dataset, indices)
    print(len(id_dataset))

    train_loader = DataLoader(id_dataset, batch_size = args.local_train_bs, shuffle=True, num_workers=0)

    return train_loader  # 只返回指定 user_id 的 DataLoader


def load_test_dataloader():

    transform = transforms.Compose([transforms.ToTensor() ,transforms.Normalize((0.5,), (0.5,))])
    
    # 使用内置函数下载mnist数据集
    test_dataset = datasets.MNIST(root='./MNIST_data', train=False, transform=transform, download=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size = args.local_test_bs, shuffle=False)

    return test_loader


def load_defense_dataloader(seed):
    torch.manual_seed(seed)
    transform = transforms.Compose([transforms.ToTensor() ,transforms.Normalize((0.5,), (0.5,))])
    
    train_dataset = datasets.MNIST(root='./MNIST_data', train=True, transform=transform, download=True)

    # 计算你需要的数据量
    data_size = len(train_dataset)
    defense_size = args.defnse_data_numbers
    
    #設定seed
    generator = torch.Generator().manual_seed(seed)

    # 使用 random_split 进行切分
    defense_dataset, _ = random_split(train_dataset, [defense_size, data_size - defense_size] ,generator=generator)
    defense_loader = DataLoader(defense_dataset, batch_size=args.local_defense_bs, shuffle=True, num_workers=2)

    return defense_loader


def load_test_defense_dataloader(seed):
    torch.manual_seed(seed)
    transform = transforms.Compose([transforms.ToTensor() ,transforms.Normalize((0.5,), (0.5,))])
 
    
    train_dataset = datasets.MNIST(root='./MNIST_data', train=True, transform=transform, download=True)

    # 计算你需要的数据量
    data_size = len(train_dataset)
    defense_size = args.test_numbers 
    
    #設定seed
    generator = torch.Generator().manual_seed(seed)

    # 使用 random_split 进行切分
    defense_dataset, _ = random_split(train_dataset, [defense_size, data_size - defense_size] ,generator=generator)
    print(len(defense_dataset))
    defense_test_loader = DataLoader(defense_dataset, batch_size=args.local_defense_bs, shuffle=False, num_workers=2)

    return defense_test_loader


def test_perturb_loader(perturbed_data):
    perturbed_data_ = perturbed_data[0].cpu()
    perturbed_target_ = perturbed_data[1].cpu()
    perturb_dataset = TensorDataset(perturbed_data_, perturbed_target_)
    perturb_loader = DataLoader(perturb_dataset, batch_size = args.local_defense_bs, shuffle=True, num_workers=2) 

    return perturb_loader


'''
def train_non_iid_with_4_shards_TEST(user_id):

    transform = transforms.Compose([transforms.ToTensor() ,transforms.Normalize((0.5,), (0.5,))])
    mnist_dataset = datasets.MNIST(root='./MNIST_data', train=True, download=True, transform=transform)

    # 建立一个空的索引列表，用于储存对应的样本索引
    indices = []

    # 为每个user_id定义对应的类别
    classes_dict = {
        0: [0, 1, 2, 3],
        1: [2, 3, 4, 5],
        2: [4, 5, 6, 7],
        3: [6, 7, 8, 9],
        4: [8, 9, 0, 1]
    }

    classes_for_user = classes_dict[user_id]

    image_count_dict = {k:0 for k in range(10)}  # to store image count per category, initially set all to 0

    for i in range(len(mnist_dataset)):
        # 如果当前样本的类别在该用户的类别列表中，则添加其索引
        if mnist_dataset.targets[i] in classes_for_user:
            indices.append(i)
            image_count_dict[int(mnist_dataset.targets[i])] += 1

    # 打印每个类别的图片数量
    print(f"User {user_id} category image counts: {dict(image_count_dict)}")

    # 使用Subset从MNIST数据集中提取出特定的样本
    id_dataset = Subset(mnist_dataset, indices)
    print(len(id_dataset))

    train_loader = DataLoader(id_dataset, batch_size = args.local_train_bs, shuffle=True, num_workers=0)

    return train_loader

'''



if __name__ == '__main__':
    # for id in range(0,5):
    #     train_non_iid_with_4_shards_TEST(id)
    pass

