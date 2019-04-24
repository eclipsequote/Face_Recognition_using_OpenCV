# -*- coding: utf-8 -*-

import cv2
import sys
import os
import time
import json

from faceTrain import Model


def faceRecognition(window_name, camera_idx):

    # 设置参数
    cv2.namedWindow(window_name)
    startTime = 0
    faceRects = []
    detection_frequency = 2  # opencv每秒图像检测频率
    grab_pic = False  # 是否抓取和保存图片
    num = 0
    catch_pic_num = 1000  # 学习样本数量
    data_path = './faceData'
    cascadeclassifier_path = './haarcascades/haarcascade_frontalface_alt2.xml'

    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    tracker_type = tracker_types[4]

    # 加载模型
    model = Model()
    model.load_model(file_path='./model/face.model.h5')

    # 配置数据源
    if camera_idx < 0:
        video_source = 'Netemo_sametemo_H265.mp4'  # 如果不使用摄像头,则调用视频文件
    else:
        video_source = camera_idx

    # 视频来源，可以来自一段已存好的视频，也可以直接来自USB摄像头
    cap = cv2.VideoCapture(video_source)

    # 获取FPS信息
    FPS = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frames = cap.get(cv2.CAP_PROP_POS_FRAMES)

    # 设置播放起始帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, FPS * 60 * startTime)

    # 告诉OpenCV使用人脸识别分类器
    classfier = cv2.CascadeClassifier(cascadeclassifier_path)

    # 识别出人脸后要画的边框的颜色，RGB格式
    color = (255, 0, 0)

    # 采样间隔帧
    frames_num = int(FPS / detection_frequency)

    # 读取人名与FaceID对照码表
    code_dict = {}
    with open('code_table.json') as table:
        code_table = json.load(table)
    # 创建码表字典
    for name, faceID in code_table:
        code_dict[faceID] = name

    while cap.isOpened():
        data_num = len([name for name in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, name))])
        ok, frame = cap.read()  # 读取一帧数据
        if not ok:
            break

        # 缩放画面
        frame = cv2.resize(frame, (800, 450), cv2.INTER_LINEAR)

        # Start timer
        timer = cv2.getTickCount()

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # 时间进度提示
        if camera_idx < 0:
            frames = cap.get(cv2.CAP_PROP_POS_FRAMES)
            progress = str(round(100*frames/count, 2))+'%'
            # Display tracker type on frame
            cv2.putText(frame, "Tracker : " + tracker_type, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Display FPS on frame
            cv2.putText(frame, "FPS : " + str(FPS) + ' ' + str(int(fps)), (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1)

            # Display progress on frame
            frames = cap.get(cv2.CAP_PROP_POS_FRAMES)
            progress = str(round(100 * frames / count, 2)) + '%'
            cv2.putText(frame, 'Schedule : ' + progress, (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(frame, time.strftime('%Y-%m-%d %H:%M:%S'), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)  # 时间提示

        if frames % frames_num == 0:

            # 将当前帧转换成灰度图像
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 人脸检测，1.2和3分别为图片缩放比例和需要检测的有效点数
            faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=5, minSize=(32, 32))

        if len(faceRects) > 0:  # 大于0则检测到人脸
            for faceRect in faceRects:  # 单独框出每一张人脸
                x, y, w, h = faceRect
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
                image = frame[y: y + h, x: x + w]
                faceID, probability = model.face_predict(image)
                if faceID > 0:
                    cv2.putText(frame, code_dict[faceID] + ' ' + str(round(probability, 3)), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)  # 文字提示

                # 将当前帧保存为图片
                if grab_pic and data_num < catch_pic_num and frames % frames_num == 0 and w >= 100 and h >= 100:
                  img_name = '%s/%d.jpg' % (data_path, int(time.time()))
                  cv2.imwrite(img_name, image)
                  num += 1

        # 显示图像
        cv2.imshow(window_name, frame)

        # Loop video
        if cap.get(cv2.CAP_PROP_POS_FRAMES) >= count - 2.:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0.)

        c = cv2.waitKey(15)
        if c & 0xFF == ord('q'):
            break

        # wait if space pressed
        elif c & 0xFF == 32:
            cv2.waitKey(0)
        elif c & 0xFF == ord('a'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) - 2 * FPS)
        elif c & 0xFF == ord('d'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + 2 * FPS)
        elif c & 0xFF == ord('z'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) - 20 * FPS)
        elif c & 0xFF == ord('c'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + 20 * FPS)

    # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Usage:%s camera_id\r\n" % (sys.argv[0]))
    else:
        faceRecognition("Face Recognition Tracking", -1)  # 第二个参数为-1时调用本地视频文件