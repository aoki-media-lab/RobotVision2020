# ライブラリのインポート
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# 実行
while True:

    # Webカメラのフレーム取得
    ret, frame = cap.read()
    cv2.imshow("camera", frame)

    """
    2-rgb2hue.pyと同じ方法で特定の色抽出
    """
    # 画像をRGBからHSVに変換
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # HSVによる上限、下限の設定　 ([Hue, Saturation, Value])
    hsvLower = np.array([0, 60, 60])  # 下限
    hsvUpper = np.array([40, 255, 255])  # 上限

    # HSVからマスクを作成
    hsv_mask = cv2.inRange(hsv, hsvLower, hsvUpper)
    cv2.imshow("hsv_mask", hsv_mask)

    # medianblurを用いてノイズ成分を除去
    blur_mask = cv2.medianBlur(hsv_mask, ksize=3)
    cv2.imshow("mask_with_medianblur", blur_mask)

    """
    ここからラベリングを行う
    """
    # ラベリング結果書き出し用に二値画像をカラー変換 (枠や座標をカラー表示したい！)
    src = cv2.cvtColor(blur_mask, cv2.COLOR_GRAY2BGR)

    # ラベリング処理
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(blur_mask)
    print(labels)

    # 領域(stats[:, 4])が3つ以上ある場合(そのうち1つは背景)だけ処理
    if len(stats[:, 4]) >= 3:
        # 面積でソート(今回は面積が上位２つの領域を利用)
        top_idx = stats[:, 4].argsort()[-3:-1]

        # 各領域において...
        for i in top_idx:

            # ターミナル上に詳細表示
            print(
                "[x0: {}, y0: {}, x幅: {}, y幅: {}, 面積: {}]".format(
                    stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3], stats[i, 4]
                )
            )

            # 領域の外接矩形の角座標を入手
            x0 = stats[i, 0]
            y0 = stats[i, 1]
            x1 = x0 + stats[i, 2]
            y1 = y0 + stats[i, 3]
            # 長方形描画 (引数 : 描画画像、 長方形の左上角、 長方形の右下角、 色(BGR)、 線の太さ)
            # 長方形以外を使いたい時はURL参照→(http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_gui/py_drawing_functions/py_drawing_functions.html)
            cv2.rectangle(src, (x0, y0), (x1, y1), (0, 0, 255), 5)

            # 領域の重心座標、サイズを表示 (引数 : 描画画像、 書き込む文字列、 書き込む座標、 フォント、 サイズ、 色、 太さ)
            cv2.putText(
                src,
                "Center X: " + str(int(centroids[i, 0])),
                (x1 - 30, y1 + 15),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                src,
                "Center Y: " + str(int(centroids[i, 1])),
                (x1 - 30, y1 + 30),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                src,
                "Size: " + str(int(stats[i, 4])),
                (x1 - 30, y1 + 45),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (0, 255, 255),
                2,
            )

    # 結果画像の表示
    cv2.imshow("output", src)

    # 終了オプション
    k = cv2.waitKey(1)
    if k == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
