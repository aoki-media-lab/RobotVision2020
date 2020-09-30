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
    hbmask = cv2.medianBlur(hsv_mask, ksize=3)
    cv2.imshow("hsv_blur_mask", hbmask)

    # ラベリング結果書き出し用に二値画像をカラー変換
    src = cv2.cvtColor(hbmask, cv2.COLOR_GRAY2BGR)

    # ラベリング処理
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(hbmask)

    # 面積でソート
    top_idx = stats[:, 4].argsort()[-3:-1]

    for i in top_idx:

        # ターミナル上に表示
        print(
            "[x0: {}, y0: {}, x幅: {}, y幅: {}, 面積: {}]".format(
                stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3], stats[i, 4]
            )
        )

        # 各オブジェクトの外接矩形を赤枠で表示
        # 長方形以外を使いたい時はURL参照→(http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_gui/py_drawing_functions/py_drawing_functions.html)
        x0 = stats[i, 0]
        y0 = stats[i, 1]
        x1 = stats[i, 0] + stats[i, 2]
        y1 = stats[i, 1] + stats[i, 3]
        cv2.rectangle(src, (x0, y0), (x1, y1), (0, 0, 255))

        # 各オブジェクトの重心座標、サイズを表示
        cv2.putText(
            src,
            "Center X: " + str(int(centroids[i, 0])),
            (x1 - 30, y1 + 15),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (0, 255, 255),
        )
        cv2.putText(
            src,
            "Center Y: " + str(int(centroids[i, 1])),
            (x1 - 30, y1 + 30),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (0, 255, 255),
        )
        cv2.putText(
            src,
            "Size: " + str(int(stats[i, 4])),
            (x1 - 30, y1 + 45),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (0, 255, 255),
        )

    # 結果画像の表示
    cv2.imshow("output", src)

    # 終了オプション
    k = cv2.waitKey(1)
    if k == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
