from models import unet
import cv2
import numpy as np
import argparse
from config import config

def predict():
    import time
    params = config()
    n_classes = params.nClasses
    input_height = params.image_height
    input_width = params.image_width

    model = unet.Unet(n_classes, input_height=input_height, input_width=input_width)

    model.load_weights("./weights/best_weights.h5")
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    if args.video:
        i = 1
        cap = cv2.VideoCapture(args.video)
        ret, frame = cap.read()
        if not ret:

            print("视频读取失败，请检查！")
            return -1
        cv2.namedWindow("org", cv2.WINDOW_NORMAL)
        cv2.namedWindow("seg", cv2.WINDOW_NORMAL)
        cv2.namedWindow("org_seg", cv2.WINDOW_NORMAL)
        cv2.namedWindow("org_seg2", cv2.WINDOW_NORMAL)
        while(1):
            t1 = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (input_width, input_height), interpolation=cv2.INTER_AREA)
            frame = np.reshape(frame, (1, input_height, input_width, 3))
            x = np.float32(frame) / 127.5 - 1

            y = model.predict(x)
            
            pr = y[0]

            pr = pr.reshape((input_height, input_width, n_classes)).argmax(axis=2)
            seg_img = np.zeros((input_height, input_width, 3))
            for c in range(n_classes):
                seg_img[:, :, 0] += ((pr[:, :] == c) * (params.colors[c][0])).astype('uint8')
                seg_img[:, :, 1] += ((pr[:, :] == c) * (params.colors[c][1])).astype('uint8')
                seg_img[:, :, 2] += ((pr[:, :] == c) * (params.colors[c][2])).astype('uint8')

            t2 = time.time()
            frame2 = frame[0].copy()
            frame2[:,:,2][seg_img[:,:,2]==255] = 255
            frame3 = frame[0].copy()
            frame3[seg_img==0] = 255
            cv2.putText(frame[0], "FPS: "+str(int(1 / (t2-t1))), (5, 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow('org', frame[0])
            cv2.imshow('seg', seg_img)
            cv2.imshow('org_seg', frame2)
            cv2.imshow('org_seg2', frame3)
            cv2.waitKey(1)
            cv2.imwrite("test/gif/org_%d.png"%i, frame[0])
            cv2.imwrite("test/gif/seg_%d.png"%i, seg_img)
            cv2.imwrite("test/gif/org_seg_%d.png"%i, frame2)
            cv2.imwrite("test/gif/org_seg2_%d.png"%i, frame3)
            i+=1

    elif args.image:
        t1 = time.time()
        x0 = cv2.imread(args.image, 1)
        x0 = cv2.resize(x0, (input_width, input_height))
        x = x0.reshape((1, input_height, input_width, 3))
        x = np.float32(x) / 127.5 - 1
        
        
        y = model.predict(x)
        pr = y[0]
        pr = pr.reshape((input_height, input_width, n_classes)).argmax(axis=2)
        seg_img = np.zeros((input_height, input_width, 3))
        for c in range(n_classes):
            seg_img[:, :, 0] += ((np.abs(pr[:, :]-c)<0.5) * (params.colors[c][0])).astype('uint8')
            seg_img[:, :, 1] += ((np.abs(pr[:, :]-c)<0.5) * (params.colors[c][1])).astype('uint8')
            seg_img[:, :, 2] += ((np.abs(pr[:, :]-c)<0.5) * (params.colors[c][2])).astype('uint8')

        t2 = time.time()
        x2 = x0.copy()
        x2[:,:,2][seg_img[:,:,2]==255] = 255
        x3 = x0.copy()
        x3[seg_img==0] = 255
        cv2.putText(x0, "cost: %.2f"%(t2-t1), (5, 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imshow('org', x0)
        cv2.imshow('seg', seg_img)
        cv2.imshow('org_seg', x2)
        cv2.imshow('org_seg2', x3)
        cv2.waitKey(0)
    else:
        print("请输入图片或视频名称！")

    return 0


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='command for training segmentation models with keras')
    parse.add_argument('--image', type=str, default=None, help='the image to predict')
    parse.add_argument('--video', type=str, default=None, help='the video to predict')
    args = parse.parse_args()
    predict()
