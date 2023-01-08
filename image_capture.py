import cv2

class Capture:
    def img_cap(count):
        videoCaptureObject = cv2.VideoCapture(0)
        _, frame = videoCaptureObject.read()
        img_name = "C:/Users/Pallavi/Desktop/auto-pro/images/NewPicture_{}.png".format(count)
        cv2.imwrite(img_name,frame)
        videoCaptureObject.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    obj = Capture
    for count in range(1,3):
        obj.img_cap(count)