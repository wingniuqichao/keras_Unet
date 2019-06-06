import numpy as np
import cv2
import random
import sys
import skimage
sys.path.append('../')

def image_enforce(image, label):
    '''
    图像增强
    '''
    if random.random()>0.2:
        # 图像亮度、对比度调整
        alpha = np.random.random()*0.6+0.4
        beta = np.random.randint(50)
        blank = np.zeros(image.shape, image.dtype)
        # dst = alpha * img + beta * blank
        dst = cv2.addWeighted(image, alpha, blank, 1-alpha, beta)
    if random.random()>0.5:
        # 图像水平翻转
        image = cv2.flip(image, 1)
        label = cv2.flip(label, 1)
    if random.random()>0.5:
        # 高斯模糊
        blurList = [3,5,7,9,11,13]
        ksize=blurList[np.random.randint(6)]
        image = cv2.GaussianBlur(image, (ksize, ksize), 0)
    return image, label


def get_batch(items, nClasses, height, width, train=True):
    x = []
    y = []
    for item in items:
        image_path = item.split(' ')[0]
        label_path = item.split(' ')[-1].strip()
        img = cv2.imread(image_path, 1)
        label_img = cv2.imread(label_path, 0)

        
        # 数据增强
        if train:
            img, label_img = image_enforce(img, label_img)

        im = np.zeros((height, width, 3), dtype='uint8')
        im[:, :, :] = 128
        lim = np.zeros((height, width, 3), dtype='uint8')

        # 图像四周扩边，使其长宽一致
        if img.shape[0] >= img.shape[1]:
            res = img.shape[0] - img.shape[1]
            if res > 0:
                if train:
                    left_res = np.random.randint(res//2+1)
                else:
                    left_res = res//2

                img = cv2.copyMakeBorder(img, 0, 0, left_res, res-left_res, cv2.BORDER_REFLECT)
                label_img = cv2.copyMakeBorder(label_img, 0, 0, left_res, res-left_res, cv2.BORDER_REFLECT)

        else:
            res = img.shape[1] - img.shape[0]
            if train:
                up_res = np.random.randint(res//2+1)
            else:
                up_res = res//2

            img = cv2.copyMakeBorder(img, up_res, res-up_res, 0, 0, cv2.BORDER_REFLECT)
            label_img = cv2.copyMakeBorder(label_img, up_res, res-up_res, 0, 0, cv2.BORDER_REFLECT)

        img = cv2.resize(img, (width, height))
        label_img = cv2.resize(label_img, (width, height))
        im = img
        lim = label_img

        seg_labels = np.zeros((height, width, nClasses))
        for c in range(nClasses):
            seg_labels[:, :, c] = (np.abs(lim-c)<0.5).astype(int)
        im = np.float32(im) / 127.5 - 1
        seg_labels = np.reshape(seg_labels, (width * height, nClasses))
        x.append(im)
        y.append(seg_labels)
    return x, y


def generator(path_file, batch_size, n_classes, input_height, input_width, train=True):
    f = open(path_file, 'r')
    items = f.readlines()
    f.close()
    if train:
        while True:
            shuffled_items = []
            index = [n for n in range(len(items))]
            random.shuffle(index)
            for i in range(len(items)):
                shuffled_items.append(items[index[i]])
            for j in range(len(items) // batch_size):
                x, y = get_batch(shuffled_items[j * batch_size:(j + 1) * batch_size],
                                 n_classes, input_height, input_width, train)
                yield np.array(x), np.array(y)
    else:
        while True:
            for j in range(len(items) // batch_size):
                x, y = get_batch(items[j * batch_size:(j + 1) * batch_size],
                                 n_classes, input_height, input_width, train)
                yield np.array(x), np.array(y)
