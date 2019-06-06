
# 配置文件
import numpy as np

class config():
    def __init__(self):
        # 图像宽度
        self.image_width = 256
        # 图像高度
        self.image_height = 256
        # 一次迭代传入多少图片，根据内存/显存调整
        self.batch_size = 16
        # 总的训练轮数
        self.epochs = 100
        # 类别数
        self.nClasses = 2
        # 样本路径
        self.train_file = './data/train2.txt'
        self.val_file = './data/val2.txt'

        self.colors = np.array([[0, 0, 0], [255, 255, 255]])