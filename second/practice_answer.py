import cv2
import numpy as np

cap = cv2.VideoCapture(0)
screenshot = False

# 実行
while True:

    ret, frame = cap.read()
    cv2.imshow("camera", frame)

    # HSVによる上限、下限の設定　 ([Hue, Saturation, Value])
    hsvLower = np.array([0, 0, 0])  # 下限
    hsvUpper = np.array([30, 150, 150])  # 上限
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # 画像をHSVに変換
    hsv_mask = cv2.inRange(hsv, hsvLower, hsvUpper)  # HSVからマスクを作成
    hbmask = cv2.medianBlur(hsv_mask, ksize=3)
    cv2.imshow("hsv_blur_mask", hbmask)

    # ラベリング結果書き出し用に二値画像をカラー変換
    src = cv2.cvtColor(hbmask, cv2.COLOR_GRAY2BGR)

    # ラベリング処理
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(hbmask)

    # 面積でソート
    top_idx = stats[:, 4].argsort()[-2]

    ball_img = cv2.imread("./image_data/ball.png")
    ball_height, ball_width = ball_img.shape[:2]
    src[
        stats[top_idx, 1] : (stats[top_idx, 1] + ball_height),
        stats[top_idx, 0] : (stats[top_idx, 0] + ball_width),
    ] = ball_img

    # 結果画像の表示
    cv2.imshow("output", src)

    # 終了オプション
    k = cv2.waitKey(1)
    if k == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
