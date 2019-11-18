import cv2
import numpy as np


def bird_view(crop_frame):
    image_h, image_w = crop_frame.shape
    src = np.float32([[0, image_h], [1000, image_h], [0, 0], [image_w, 0]])     # 1207
    dst = np.float32([[520, image_h], [780, image_h], [0, 0], [image_w, 0]])    # 568, 711, (420, 720)
    M = cv2.getPerspectiveTransform(src, dst)
    bird_img = cv2.warpPerspective(crop_frame, M, (image_w, image_h))
    return bird_img


def cam_module():
    cap = cv2.VideoCapture(0)
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
        warp = bird_view(normalized_img)
        # cv2.imshow("warp", warp)
        # cv2.imshow("norm", normalized_img)
        low_threshold = 50
        high_threshold = 150
        canny = cv2.Canny(warp, low_threshold, high_threshold)
        cv2.imshow("filtered", canny)
        line_detect = cv2.HoughLines(canny, 1, np.pi/160, 200)

        # for r, theta in line_detect[0]:
        #     a = np.cos(theta)
        #     b = np.sin(theta)
        #     x0 = a*r
        #     y0 = b*r
        #     x1 = int(x0 + 1000*(-b))
        #     y1 = int(y0 + 1000*(a))
        #     x2 = int(x0 - 1000*(-b))
        #     y2 = int(y0 - 1000*(a))
        #     cv2.line(crop_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        if line_detect is not None:
            for i in range(0, len(line_detect)):
                rho = line_detect[i][0][0]
                theta = line_detect[i][0][1]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                cv2.line(crop_img, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

        # out.write(frame)
        cv2.imshow("final", crop_img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # The following frees up resources and closes all windows
    cap.release()
    # out.release()
    cv2.destroyAllWindows()


cam_module()
