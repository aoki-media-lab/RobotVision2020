# ライブラリのインポート
import copy

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# アプリ用のスタジアム、ボール画像を読みこみ
ret, frame = cap.read()
ball_img = cv2.imread("./image_data/ball.png")
stadium_img = cv2.imread("./image_data/stadium.png")

# Webカメラの画面の大きさにスタジアムを合わせる
stadium_img = cv2.resize(stadium_img, (frame.shape[1], frame.shape[0]))

# ボールの高さ、幅の[半分](半分だから注意！！)
# (注意!)今回ボールの大きさが H:198、W:200と両方偶数のためこれで良いが、奇数の場合は工夫が必要
ball_h, ball_w = ball_img.shape[0] // 2, ball_img.shape[1] // 2

# ボールの初期位置（中心座標)をスタジアムの中心に設定
idx_h = stadium_img.shape[0] // 2
idx_w = stadium_img.shape[1] // 2

# はじめボールは中央に配置
stadium = copy.deepcopy(stadium_img)
stadium[
    (idx_h - ball_h) : (idx_h + ball_h), (idx_w - ball_w) : (idx_w + ball_w)
] = ball_img


# 実行
while True:

    # Webカメラのフレーム取得
    ret, frame = cap.read()
    cv2.imshow("camera", frame)

    """
    4-labeling.pyと同じ方法でラベリング
    ただし今回は最も大きい領域(4-labeling.pyでは上位２つ)のみ利用
    """
    # 画像をRGBからHSVに変換
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # HSVによる上限、下限の設定　 ([Hue, Saturation, Value])
    hsvLower = np.array([0, 30, 30])  # 下限
    hsvUpper = np.array([30, 150, 150])  # 上限

    # HSVからマスクを作成
    hsv_mask = cv2.inRange(hsv, hsvLower, hsvUpper)

    # medianblurを用いてノイズ成分を除去
    blur_mask = cv2.medianBlur(hsv_mask, ksize=3)

    # ラベリング結果書き出し用に二値画像をカラー変換
    src = cv2.cvtColor(blur_mask, cv2.COLOR_GRAY2BGR)

    # ラベリング処理
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(blur_mask)

    # 領域(stats[:, 4])が２つ以上ある場合(そのうち1つは背景)だけ処理
    if nlabels >= 2:
        # 面積でソート、　今回は最も大きい領域１つだけ利用
        idx = stats[:, 4].argsort()[-2]

        # 領域の外接矩形の角座標を入手
        x0 = stats[idx, 0]
        y0 = stats[idx, 1]
        x1 = x0 + stats[idx, 2]
        y1 = y0 + stats[idx, 3]

        # 長方形描画 (引数 : 描画画像、 長方形の左上角、 長方形の右下角、 色(BGR)、 線の太さ)
        cv2.rectangle(src, (x0, y0), (x1, y1), (0, 0, 255), 5)

        # 領域の重心座標、サイズを表示 (引数 : 描画画像、 書き込む文字列、 書き込む座標、 フォント、 サイズ、 色、 太さ)
        cv2.putText(
            src,
            "Center X: " + str(int(centroids[idx, 0])),
            (x1 - 30, y1 + 15),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            src,
            "Center Y: " + str(int(centroids[idx, 1])),
            (x1 - 30, y1 + 30),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            src,
            "Size: " + str(int(stats[idx, 4])),
            (x1 - 30, y1 + 45),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (0, 255, 255),
            2,
        )

        """
        ここから少し工夫をしてアプリ化
        ボールをスタジアムに配置する
        """
        # ボールの配置位置の中心座標をラベリング結果の重心座標で指定
        idx_h = int(centroids[idx, 1])
        idx_w = int(centroids[idx, 0])

        # ボールがスタジアムからはみ出す時、位置を調整
        if idx_h < ball_h:
            idx_h = ball_h
        elif idx_h >= stadium_img.shape[0] - ball_h:
            idx_h = stadium_img.shape[0] - ball_h - 1
        if idx_w < ball_w:
            idx_w = ball_w
        elif idx_w >= stadium_img.shape[1] - ball_w:
            idx_w = stadium_img.shape[1] - ball_w - 1

        # ボールの再配置
        stadium = copy.deepcopy(stadium_img)
        stadium[
            (idx_h - ball_h) : (idx_h + ball_h), (idx_w - ball_w) : (idx_w + ball_w)
        ] = ball_img

    # 結果画像の表示
    cv2.imshow("labeling", src)
    cv2.imshow("output", stadium)

    # 終了オプション
    k = cv2.waitKey(1)
    if k == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
