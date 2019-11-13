import cv2


frame = cv2.imread("dahedra_armor.png")
cuMat = cv2.cuda_GpuMat()
cuMat.upload(frame)

# cv2.imshow('window', frame)
