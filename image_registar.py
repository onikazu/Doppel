import cv2
import time
import datetime


def main():
    cap = cv2.VideoCapture(0)  # 引数はカメラのデバイス番号
    name = input("あなたの名前を教えてください")
    t = datetime.datetime.now()
    print("{}さんですね。これからあなたの写真を登録します。".format(name))

    print("10秒後に写真を撮影します")
    time.sleep(7)
    print("3秒前")
    time.sleep(3)
    ret, frame = cap.read()

    cv2.imshow("frame", frame)
    k = cv2.waitKey(1)
    judge = input("写真の撮影が完了しました. 保存してよろしければ y と入力してください")

    if judge != "y":
        return

    cv2.imwrite("add_database/{0}_{1:%Y%m%d}.jpg".format(name, t), frame)
    cap.release()
    cv2.destroyAllWindows()
    print("保存が完了しました")

if __name__ == '__main__':
    main()
