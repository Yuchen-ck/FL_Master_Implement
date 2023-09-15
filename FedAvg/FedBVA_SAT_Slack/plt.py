import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    SFAT_robust =[0.10627, 0.01979, 0.04112, 0.08697, 0.05469, 0.02639, 0.02284, 0.02871, 0.02526, 0.03472, 0.01954, 0.01192, 0.00985, 0.00756, 0.00789, 0.00513, 0.00684, 0.00577, 0.0065, 0.00469]
    # Scaffold = [0.915, 0.91, 0.931, 0.937, 0.938, 0.939, 0.938, 0.938, 0.938, 0.939, 0.938, 0.939, 0.939, 0.939, 0.938, 0.939, 0.938, 0.939, 0.938 ,0.939]

    # fedAvg = fedAvg[:20]
    # robust = robust[:20]
    # robust = sorted(robust)
    y =  list(range(len(SFAT_robust)))
    fig = plt.figure()
    # plt.plot(y, fedAvg ,label='FedAvg',color ='#66B3FF')
    plt.plot(y, SFAT_robust ,label='SFAT-robust')
    # plt.plot(y, Scaffold ,label='Scaffold')

    
    plt.title('Non-IID')
    plt.xlabel('communication rounds')
    plt.ylabel('R2_score')
    plt.legend()
    
    #限制刻度的範圍
    # plt.ylim(0,100)

    # 找到 FedAvg 和 robust 列表中的最大值
    # fedAvg_max = max(fedAvg)
    # robust_max = max(SFAT_robust)

    # 在最高點的位置上畫出點，並添加標籤
    # plt.plot(fedAvg.index(fedAvg_max), fedAvg_max, 'o', color='#0080FF', markersize=4)
    # plt.plot(robust.index(robust_max), robust_max, 'o', color='#FF0000', markersize=4)

    # plt.text(fedAvg.index(fedAvg_max), fedAvg_max, str(round(fedAvg_max,3)))
    # plt.text(robust.index(robust_max), robust_max, str(round(robust_max,3)))


    plt.savefig('plot_with_cround.png')
    plt.show()