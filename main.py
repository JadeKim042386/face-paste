import cv2
from facenet_pytorch import MTCNN
import torch
import numpy as np

# -- device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# -- model
mtcnn = MTCNN(keep_all=True, device=device)

# -- webcam
cap = cv2.VideoCapture(0)

# -- load img
bear_img = cv2.imread('bear.png')
bear_img = cv2.resize(bear_img, dsize=(512, 512))

while cap.isOpened():
    ret, img = cap.read() # current frame

    if not ret:
        break

    boxes, _, landmarks = mtcnn.detect(img, landmarks=True)
    
    if isinstance(landmarks, np.ndarray) and len(landmarks) > 0:
        for landmark in landmarks: # [[[x, y]]]
            # 0, 1 : eyes
            len_x = 15
            len_y = 10

            cnt_x1 = int(landmark[0][0])
            cnt_y1 = int(landmark[0][1])
            left_eye_img = img[cnt_y1 - len_y:cnt_y1 + len_y, cnt_x1 - len_x:cnt_x1 + len_x].copy()
            
            cnt_x2 = int(landmark[1][0])
            cnt_y2 = int(landmark[1][1])
            right_eye_img = img[cnt_y2 - len_y:cnt_y2 + len_y, cnt_x2 - len_x:cnt_x2 + len_x].copy()

            left_eye_img = cv2.resize(left_eye_img, dsize=(150, 100))
            right_eye_img = cv2.resize(right_eye_img, dsize=(150, 100))

            # 3, 4 : mouth
            len_x2 = 5
            len_y2 = 10

            cnt_x3 = int(landmark[3][0]) # left
            cnt_y3 = int(landmark[3][1]) # y-axis
            cnt_x4 = int(landmark[4][0]) # right

            mouth_img = img[cnt_y3 - len_y2:cnt_y3 + len_y2, cnt_x3 - len_x2:cnt_x4 + len_x2].copy()
            mouth_img = cv2.resize(mouth_img, dsize=(200, 100))

            # -- paste
            result = bear_img.copy()

            result = cv2.seamlessClone(
                left_eye_img,
                result,
                np.full(left_eye_img.shape[:2], 255, left_eye_img.dtype),
                (150, 200),
                cv2.NORMAL_CLONE
            )
            result = cv2.seamlessClone(
                right_eye_img,
                result,
                np.full(right_eye_img.shape[:2], 255, right_eye_img.dtype),
                (350, 200),
                cv2.NORMAL_CLONE
            )
            result = cv2.seamlessClone(
                mouth_img,
                result,
                np.full(mouth_img.shape[:2], 255, mouth_img.dtype),
                (254, 400),
                cv2.NORMAL_CLONE
            )

    else:
        result = img.copy()

    # -- show
    cv2.imshow('result', result)

    if cv2.waitKey(1) == ord('q'):
        break