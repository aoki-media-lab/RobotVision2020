import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# 実行
while True:

    ret, frame = cap.read()
    cv2.imshow("camera", frame)

    # HSVによる上限、下限の設定　 ([Hue, Saturation, Value])
    hsvLower = np.array([0, 0, 0])  # 下限
    hsvUpper = np.array([30, 150, 150])  # 上限
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # 画像をHSVに変換
    hsv_mask = cv2.inRange(hsv, hsvLower, hsvUpper)  # HSVからマスクを作成
    blur_mask = cv2.medianBlur(hsv_mask, ksize=3)
    cv2.imshow("hsv_mask", hsv_mask)
    cv2.imshow("mask_with_medianblur", blur_mask)

    # 終了オプション
    k = cv2.waitKey(1)
    if k == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
