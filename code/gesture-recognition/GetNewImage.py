import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2 as cv
import TestGesture
img_roi_y = 30
img_roi_x = 200
img_roi_height = 300
img_roi_width = 300
capture = cv.VideoCapture(0)
index = 1
while True:
    ret, frame = capture.read()
    if ret is True:
        img_roi = frame[img_roi_y:(img_roi_y + img_roi_height), img_roi_x:(img_roi_x + img_roi_width)]
        cv.imshow("frame", img_roi)
        index += 1
        if index % 20 == 0:   # 每5帧保存一次图像
            path = "./data/testImage/1/"
            os.system("rm data/testImage/1/*")
            name = "test"  # 给录制的手势命名
            cv.imwrite(path + name + '.jpg', img_roi)
            print(name)
        c = cv.waitKey(50)  # 每50ms判断一下键盘的触发。  0则为无限等待。
        if c == 27:  # 在ASCII码中27表示ESC键，ord函数可以将字符转换为ASCII码。
            break
        if index == 20:
            break
    else:
        break

cv.destroyAllWindows()
capture.release()


gesture_num = TestGesture.evaluate_one_image()
if gesture_num == 1:
    gesture_action = "1"
    print("剪刀")
elif gesture_num == 2:
    gesture_action = "2"
    print("石头")
elif gesture_num == 3:
    gesture_action = "3"
    print("布")
elif gesture_num == 4:
    gesture_action = "4"
    print("OK")
elif gesture_num == 5:
    gesture_action = "5"
    print("赞")