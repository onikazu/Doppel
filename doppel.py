import cv2
import subprocess
import time
import datetime
import numpy as np
import dlib
import pickle
import glob
import face_recognition
import faiss
import slackweb
import random

WEB_HOOK_URL = "https://hooks.slack.com/services/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"

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

def main():
    cap = cv2.VideoCapture(0)  # 引数はカメラのデバイス番号
    print("顧客新規登録にはnを、リピーターか確かめるにはcを、終了する場合はqを入力してください")
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.imshow('FaceMandara', frame)
        k = cv2.waitKey(1)&0xff
        if k == ord('q'):
            print("released!")
            break
        if k == ord('n'):
            register(frame)
            print("顧客新規登録にはnを、リピーターか確かめるにはcを、終了する場合はqを入力してください")
        if k == ord('c'):
            check_repeater(frame)
            print("顧客新規登録にはnを、リピーターか確かめるにはcを、終了する場合はqを入力してください")
    cap.release()
    cv2.destroyAllWindows()
    print("release camera!!!")

def register(frame):
    # 反転を戻す
    order = input("登録フェイズに移行します。よろしいですか？[y/n]")
    if order == "n":
        return
    frame = cv2.flip(frame, 1)
    frame = np.array(frame)
    flag, rects = _check_face(frame)
    if not flag:
        print("入力画像から人物を認識できませんでした")
        return
    name = input("名前を入力してください")
    t = datetime.datetime.now()
    cv2.imwrite("add_database/{0}_{1:%Y%m%d}.jpg".format(name, t), frame)
    _add_database()
    print("登録が完了しました")
    return

def check_repeater(frame):
    # 反転を戻す
    frame = cv2.flip(frame, 1)
    frame = np.array(frame)
    flag, rect = _check_face(frame)
    if not flag:
        return
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
        register(frame)
        return
    else:
        print("あなたは{}ですね".format(recommend_path))
        _check_carrier(recommend_path)
        while True:
            order = input("照合を再開する場合はyと入力してください")
            if order == "y":
                return


def _check_customer(similar_paths):
    flag = False
    recommend_path = ""
    for path in similar_paths:
        print("path", path)
        if path.find('_') > -1:
            flag = True
            recommend_path = path
            break
    return flag, recommend_path

def _check_carrier(path):
    target_name = "kazuki"
    name = path.split("_")[0]
    if name == target_name:
        _send_slack(target_name)


def _check_face(frame):
    # 顔認識機のインスタンス
    detector = dlib.get_frontal_face_detector()
    # 顔データ(人数分)
    rects = detector(frame, 1)
    if not rects:
        print("人物が読み込めません")
        return False
    if len(rects) != 1:
        print("画面に複数人数写り込んでしまっています")
        return False
    rect = rects[0]
    return True, rect

def _add_database():
    big_vector_images = {}
    # ピックル化された辞書データの読み込み
    with open('big_data.pickle', mode='rb') as f:
        big_vector_images = pickle.load(f)

    add_image_path = "./add_database"
    add_images = glob.glob(add_image_path + "/*.jpg")

    # 辞書型のデータを作る
    for image_file in add_images:
        image = face_recognition.load_image_file(image_file)
        # 顔認識
        detector = dlib.get_frontal_face_detector()
        rects = detector(image, 1)
        # 顔認識してい無いとき
        if not rects:
            continue
        face_encoding = face_recognition.face_encodings(image)[0]
        print(len(face_encoding.tolist()))
        big_vector_images[image_file.split("/")[-1]] = face_encoding.tolist()

    # vector_images is like {"face0.jpg":[0.11,.....], }
    # with open('data.pickle', mode='wb') as f:
    with open('big_data.pickle', mode='wb') as f:
        pickle.dump(big_vector_images, f)

    # add_imagesをbig_databaseに移す。
    for image_file in add_images:
        image_file = image_file.split("/")[-1]
        cmd = "mv ./add_database/{0} ./big_database/{0}".format(image_file)
        cmd = cmd.split(" ")
        subprocess.run(cmd)
    print("{}をデータベースに追加しました".format(add_images[0]))

def _send_slack(name):
    slack = slackweb.Slack(url=WEB_HOOK_URL)
    slack.notify(text="{}がエントランスで待っています。対応してください".format(name))

if __name__ == "__main__":
    main()
