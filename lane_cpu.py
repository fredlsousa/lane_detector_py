import cv2
import numpy as np


def bird_view(frame, crop_frame):
    image_h, image_w, _ = frame.shape
    src = np.float32([[0, image_h], [1207, image_h], [0, 0], [image_w, 0]])
    dst = np.float32([[568, image_h], [711, image_h], [0, 0], [image_w, 0]])
    M = cv2.getPerspectiveTransform(src, dst)
    bird_img = cv2.warpPerspective(crop_frame, M, (image_w, image_h))
    return bird_img


def cam_module():
    cap = cv2.VideoCapture("video_1.avi")
    # out = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc(*"H264"), 30, (1280, 960))

    while cap.isOpened():
        ret, frame = cap.read()
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        # cv2.imshow("gray", gray_img)

        img_shape = gray_img.shape
        crop_img = gray_img[480:960, 0:1280]

        median = cv2.medianBlur(crop_img, 7)
        # cv2.imshow("gauss", median)
        normalized_img = median
        cv2.normalize(median, normalized_img, 0, 255, cv2.NORM_MINMAX)
        warp = bird_view(frame, normalized_img)
        cv2.imshow("warp", warp)
        # cv2.imshow("norm", normalized_img)
        low_threshold = 50
        high_threshold = 150
        canny = cv2.Canny(warp, low_threshold, high_threshold)
        cv2.imshow("filtered", canny)


        # out.write(frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # The following frees up resources and closes all windows
    cap.release()
    # out.release()
    cv2.destroyAllWindows()


cam_module()
