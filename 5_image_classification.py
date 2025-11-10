import cv2
import jetson.inference
import jetson.utils

img = cv2.read('cat.jpg')
rgba = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
imgCuda = jetson.utils.cudaFromNumpy(rgba)
net = jetson.inference.imageNet('googlenet')
classID, confidence = net.Classify(imgCuda)
classDesc = net.GetClassDesc(classID)

cv2.putText(img, classDesc, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

print("Image is recognized as '{:s}' with {:.2f} % confidence".format(classDesc, confidence * 100))
cv2.imshow('img', img)
cv2.waitKey(0) 