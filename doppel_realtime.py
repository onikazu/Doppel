import subprocess
import time
import datetime
import pickle
import glob
import random
from multiprocessing import Manager, Process
import multiprocessing as mp
import configparser

import cv2
import numpy as np
import dlib
import face_recognition
import faiss
import slackweb


config = configparser.SafeConfigParser()
config.read('slack.ini')
WEB_HOOK_URL = config.get('web_hook', 'url1')

# faiss を用いた高速探索（indexing）
print("start indexing")
datas = {}
with open('big_data.pickle', mode='rb') as f:
    datas = pickle.load(f)
# databese配列の作成
face_image_names = []
face_vectors = []
for k in datas:
    face_image_names.append(k)
    face_vectors.append(datas[k])
face_vectors = np.array(face_vectors).astype("float32")

# faissを用いたPQ
nlist = 100
m = 8
k = 8  # 類似顔7こほしいのでk=8
d = 128  # 顔特徴ベクトルの次元数
quantizer = faiss.IndexFlatL2(d)  # this remains the same
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
index.train(face_vectors)
index.add(face_vectors)
print("indexing is end")
print("finish indexing")


def check_carrier(shared_frame, lock):
    while True:
        # shared_frameの参照
        if not shared_frame:
            continue
        #lock.acquire()
        frame = shared_frame[0]
        #lock.release()
        frame = np.array(frame)
        flag, rect = _check_face(frame)
        if not flag:
            print("顔が認識できません")
            continue
        dst = frame[rect.top():rect.bottom(), rect.left():rect.right()]
        try:
            target_image_encoded = face_recognition.face_encodings(dst)[0]
        except IndexError:
            pass
        target_vector = np.array(list(target_image_encoded)).astype("float32")
        target_vector.resize((1, 128))
        # ここに入る
        similar_paths = []
        D, I = index.search(target_vector, k)
        for i in range(1, len(I[0])):
            similar_paths.append(face_image_names[I[0][i]])
        flag, recommend_path = _check_customer(similar_paths)
        print(similar_paths)
        if not flag:
            print("データベースと照合した結果、同一人物は見つかりませんでした")
            continue
        else:
            print("あなたは{}ですね".format(recommend_path))
            _check_carrier(recommend_path)

def _check_customer(similar_paths):
    flag = False
    recommend_path = ""
    for path in similar_paths:
        if path.find('_') > -1:
            flag = True
            recommend_path = path
            break
    return flag, recommend_path

def _check_carrier(path):
    name = path.split("_")[0]
    if name == "sagawa":
        _send_slack(name="佐川急便")
    elif name == "mail":
        _send_slack(name="郵便配達員")
    elif name == "kuroneko":
        _send_slack(name="クロネコヤマト便")

def _send_slack(name):
    slack = slackweb.Slack(url=WEB_HOOK_URL)
    slack.notify(text="{}がエントランスで待っています。対応してください".format(name))
    print("{}がエントランスで待っています。対応してください".format(name))
    print("10秒間のスリープモードに入ります")
    time.sleep(10)

def _check_face(frame):
    # 顔認識機のインスタンス
    detector = dlib.get_frontal_face_detector()
    # 顔データ(人数分)
    rects = detector(frame, 1)
    if not rects:
        return False, 0
    if len(rects) != 1:
        print("画面に複数人数写り込んでしまっています")
        return False, 0
    rect = rects[0]
    return True, rect



# カメラ動作部分
if __name__ == '__main__':
    with Manager() as manager:
    # lockの作成
        lock = manager.Lock()
        shared_frame = manager.list()
        ctx = mp.get_context("spawn")
        recommend_process = ctx.Process(target=check_carrier, args=[shared_frame, lock], name="check_carrier")
        # プロセスの開始
        recommend_process.start()

        # ビデオオブジェクト
        cap = cv2.VideoCapture(0)
        # 撮影の開始
        while True:
            ret, frame = cap.read()
            k = cv2.waitKey(1)
            cv2.imshow("Doppel", frame)
            k = cv2.waitKey(1)&0xff
            if k == ord('q'):
                print("released!")
                break
            # フレームの更新
            # 配列への変換/共有メモリへの代入
            if shared_frame[:] == []:
                shared_frame.append(list(frame))
            else:
                shared_frame[0] = list(frame)
        cap.release()
        cv2.destroyAllWindows()
        print("release camera!!!")
