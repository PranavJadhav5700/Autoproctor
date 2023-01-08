import cv2

objects = []


def detect_img(image):
    img = cv2.imread(image)
    # className= ['person','laptop','cellphone']
    classFile = 'files/coco.names'
    with open(classFile,'rt') as f:
        className = f.read().rstrip('\n').split('\n')
    configPath = 'files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = 'files/frozen_inference_graph.pb'
    
    net = cv2.dnn_DetectionModel(weightsPath,configPath)
    net.setInputSize(320,320)
    net.setInputScale(1.0/ 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    classIds, confs, bbox = net.detect(img,confThreshold=0.5)
    # print(classIds,bbox)
    for classId, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
        cv2.rectangle(img, box, color=(0,255,0), thickness=2)
        # print("Box: ",box)
        objects.append(className[classId-1].upper())
        cv2.putText(img,className[classId-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        # cv2.imwrite("Cut.png",img[box[0]-10:box[0] + box[2], box[1]-30:(box[1]) + box[-1]])

    # cv2.imshow("Output",img)
    # print(cut)

    cv2.waitKey(0)


def calc():
	detect_img('images/NewPicture_1.png')
	detect_img("images/NewPicture_2.png")
	

def execute():
    calc()
    # print(objects)
    # return(objects)


execute()
print(objects)
