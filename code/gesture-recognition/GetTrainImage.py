import cv2 as cv

img_roi_y = 30
img_roi_x = 200
img_roi_height = 300
img_roi_width = 300
capture = cv.VideoCapture(0)
index = 1
num = 0
while True:
    ret, frame = capture.read()
    if ret is True:
        img_roi = frame[img_roi_y:(img_roi_y + img_roi_height), img_roi_x:(img_roi_x + img_roi_width)]
        cv.imshow("frame", img_roi)
        index += 1
        if index % 1 == 0:
            num += 1
            path = "./data/train/"
            name = "gesture_5."+str(num)  # 给录制的手势命名
            cv.imwrite(path + name + '.jpg', img_roi)
            print(name)
        c = cv.waitKey(50)  # 每50ms判断一下键盘的触发。  0则为无限等待。
        if c == 27:  # 在ASCII码中27表示ESC键，ord函数可以将字符转换为ASCII码。
            break
        if index == 1301:
            break
    else:
        break

cv.destroyAllWindows()
capture.release()
