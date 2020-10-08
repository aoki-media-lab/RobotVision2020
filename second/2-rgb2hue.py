# ライブラリのインポート
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# 実行
while True:

    # Webカメラのフレーム取得
    ret, frame = cap.read()
    cv2.imshow("camera", frame)

    # 画像をRGBからHSVに変換
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # HSVによる上限、下限の設定　 ([Hue, Saturation, Value])
    hsvLower = np.array([0, 0, 0])  # 下限
    hsvUpper = np.array([150, 150, 150])  # 上限

    # HSVからマスクを作成
    hsv_mask = cv2.inRange(hsv, hsvLower, hsvUpper)
    cv2.imshow("hsv_mask", hsv_mask)

    # medianblurを用いてノイズ成分を除去
    blur_mask = cv2.medianBlur(hsv_mask, ksize=3)
    cv2.imshow("mask_with_medianblur", blur_mask)

    # 終了オプション
    k = cv2.waitKey(1)
    if k == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
