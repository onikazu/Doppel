import dlib

import face_recognition
import json
import sys
import glob
import pickle
import subprocess

# trimmed_image_path = "./database/img_align_celeb"


big_vector_images = {}
# ピックル化された辞書データの読み込み
with open('big_data.pickle', mode='rb') as f:
    big_vector_images = pickle.load(f)

add_image_path = "./add_database"
add_images = glob.glob(add_image_path + "/*.jpg")


data_num = 0
# 辞書型のデータを作る
for image_file in add_images:
    image = face_recognition.load_image_file(image_file)
    # 顔認識
    detector = dlib.get_frontal_face_detector()
    rects = detector(image, 1)
    print(data_num)

    # 顔認識してい無いとき
    if not rects:
        print("{} can not be recognized".format(image_file))
        continue
    face_encoding = face_recognition.face_encodings(image)[0]
    print(len(face_encoding.tolist()))
    big_vector_images[image_file.split("/")[-1]] = face_encoding.tolist()
    data_num += 1

# vector_images is like {"face0.jpg":[0.11,.....], }
# with open('data.pickle', mode='wb') as f:
with open('big_data.pickle', mode='wb') as f:
    pickle.dump(big_vector_images, f)

# add_imagesをbig_databaseに移す。
for image_file in add_images:
    print(image_file)
    image_file = image_file.split("/")[-1]
    print(image_file)
    cmd = "mv ./add_database/{0} ./big_database/{0}".format(image_file)
    cmd = cmd.split(" ")
    subprocess.run(cmd)
print("finished")
