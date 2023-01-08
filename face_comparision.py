from deepface import DeepFace
import cv2
# import matplotlib.pyplot as plt

class Compare:
	def execute():
		img1 = cv2.imread("images/NewPicture_1.png")
		img2 = cv2.imread("images/NewPicture_2.png")
		result = DeepFace.verify(img1, img2, enforce_detection = False)
		return(result)


a = Compare.execute()
print(a)