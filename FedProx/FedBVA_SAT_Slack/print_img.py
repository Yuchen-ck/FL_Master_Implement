import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.transforms import Resize
import os 
from args import argparse_
args = argparse_()

robust_img_path = args.robust_img_folder

pgd_img_path = args.pgd_img_folder


def create_folder(folder_path):

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    print("Creating the folder is successful.")


def draw(data_loader , t ,save_folder):
    
    create_folder(save_folder)   

    # 從 robust_test_loader 取得第一批次的圖片與標籤
    images, labels = next(iter(data_loader))

    # 初始化圖片大小調整函數
    resize = Resize((64, 64))

    # 創建一個 4x4 的 subplot
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))

    # 遍歷前 16 張圖片
    for i in range(16):
        img = images[i]
        img = transforms.ToPILImage()(img)  # 將 PyTorch 張量轉換成 PIL 圖片
        img = resize(img)  # 將圖片大小調整為 64x64

        # 在 subplot 中的相應位置繪製圖片
        ax = axes[i // 4, i % 4]
        ax.imshow(img, cmap='gray')
        ax.axis('off')  # 隱藏座標軸
        ax.set_title(f'Class: {labels[i]}', fontsize=20)  # Set title with class label

    # 儲存整個 subplot 為一張圖片
    plt.savefig(save_folder + f'./bva_img_{t}.png')
    plt.close(fig)  # 關閉圖片，避免使用過多記憶體



def draw_pgd(data_loader , t ,save_folder):
    print("draw_pgd!!!!!")
    create_folder(save_folder)   

    # 從 robust_test_loader 取得第一批次的圖片與標籤
    images, labels = next(iter(data_loader))

    # 初始化圖片大小調整函數
    resize = Resize((64, 64))

    # 創建一個 4x4 的 subplot
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))

    # 遍歷前 16 張圖片
    for i in range(16):
        img = images[i]
        img = transforms.ToPILImage()(img)  # 將 PyTorch 張量轉換成 PIL 圖片
        img = resize(img)  # 將圖片大小調整為 64x64

        # 在 subplot 中的相應位置繪製圖片
        ax = axes[i // 4, i % 4]
        ax.imshow(img, cmap='gray')
        ax.axis('off')  # 隱藏座標軸
        ax.set_title(f'Class: {labels[i]}', fontsize=20)  # Set title with class label

    # 儲存整個 subplot 為一張圖片
    plt.savefig(save_folder + f'./pgd_img_{t}.png')
    plt.close(fig)  # 關閉圖片，避免使用過多記憶體