import cv2
import numpy as np

# install ipython
# ctrl + r  - shows the command histroy on ipython and linux terminal
# ctrl + w - delete every word in terminal
# help(<python command>) - shows the "documentation" of a class


def color_filter(frame, hsv_mat, gray_mat, lower, upper):
    mask_cw = cv2.cuda_GpuMat()
    mask_cw_im = cv2.cuda_GpuMat()
    lower_white = np.array([0, 0, 0], dtype="uint8")
    upper_white = np.array([0, 0, 255], dtype="uint8")
    mask_range = in_range(frame, hsv_mat, lower, upper)
    mask_white = in_range(frame, gray_mat, lower_white, upper_white)
    mask_cw = cv2.cuda.bitwise_or(mask_white, mask_range)
    mask_cw_im = cv2.cuda.bitwise_and(gray_mat, mask_cw)
    return mask_cw_im


def in_range(frame, img_mat, lower, upper):
    h, w, _ = frame.shape
    mat_parts_low = [cv2.cuda_GpuMat(h, w, cv2.CV_8UC1) for i in range(3)]
    mat_parts_high = [cv2.cuda_GpuMat(h, w, cv2.CV_8UC1) for i in range(3)]
    mat_parts = [cv2.cuda_GpuMat(h, w, cv2.CV_8UC1) for i in range(3)]
    cv2.cuda.split(img_mat, dst=mat_parts)

    for i in range(3):
        cv2.cuda.threshold(mat_parts[i], lower[i], 255, cv2.THRESH_BINARY, mat_parts_low[i])
        cv2.cuda.threshold(mat_parts[i], upper[i], 255, cv2.THRESH_BINARY, mat_parts_high[i])
        cv2.cuda.bitwise_and(mat_parts_high[i], mat_parts_low[i], mat_parts[i])

    tmp1 = cv2.cuda_GpuMat()
    final_res = cv2.cuda_GpuMat()
    tmp1 = cv2.cuda.bitwise_and(mat_parts[0], mat_parts[1])   # bug here
    final_res = cv2.cuda.bitwise_and(tmp1, mat_parts[2])
    return final_res


def main_module():
    # cap = cv2.VideoCapture(0)
    img_mat = cv2.cuda_GpuMat()

    #while(cap.isOpened()):
    # ret, frame = cap.read()

    lower_black = np.array([0, 0, 0], dtype="uint8")
    upper_black = np.array([230, 255, 65], dtype="uint8")

    frame = cv2.imread("dahedra_armor.png")
    cv2.imshow("frame", frame)
    img_mat.upload(frame)
    gray_img = cv2.cuda.cvtColor(img_mat, cv2.COLOR_RGB2GRAY)
    hsv_mat = cv2.cuda.cvtColor(img_mat, cv2.COLOR_RGB2HSV)
    gauss = cv2.cuda.createGaussianFilter(cv2.CV_8UC3, -1, (3, 3), 16)

    filtered_col = color_filter(frame, hsv_mat, gray_img, lower_black, upper_black)
    fil = filtered_col.download()
    # cv2.imshow("fil", fil)
    gauss_filtered = gauss.apply(filtered_col)
    # gauss_filtered = gauss.apply(hsv_mat)

    detector = cv2.cuda.createCannyEdgeDetector(50, 150)
    detector.detect(gauss_filtered)
    canny = detector.download()
    cv2.imshow("Canny", canny)

    # if cv2.waitKey(10) & 0xFF == ord('q'):
    #     break

    # The following frees up resources and closes all windows
    # cap.release()
    # cv2.destroyAllWindows()

main_module()
