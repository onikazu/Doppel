import cv2


cap = cv2.VideoCapture(0)  # 引数はカメラのデバイス番号

while True:
    ret, frame = cap.read()
    print(frame.shape)
