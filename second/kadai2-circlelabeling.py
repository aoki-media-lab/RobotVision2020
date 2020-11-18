# ライブラリのインポート
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# 色の範囲
# HSVRange["blue"]["lower"]で値を取り出せる
HSVRange = {
    "blue": {"lower": np.array([100, 50, 50]), "upper": np.array([120, 255, 255])},
    "green": {"lower": np.array([50, 50, 50]), "upper": np.array([60, 255, 255])},
    "pink": {"lower": np.array([160, 50, 50]), "upper": np.array([170, 255, 255])},
}

# カラーのリスト作成
Colors = ["blue", "green", "pink"]

# 実行
while True:
    # -----------以下記述-----------

    # Webカメラのフレーム取得
    ret, frame = cap.read()
    cv2.imshow("camera", frame)

    """
    2-rgb2hue.pyと同じ方法で特定の色抽出
    """
    # 画像をRGBからHSVに変換
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 各色のマスクをdictで管理
    hsv_masks = {}

    for color in Colors:
        # HSVからマスクを作成
        hsv_mask = cv2.inRange(hsv, HSVRange[color]["lower"], HSVRange[color]["upper"])
        hsv_masks[color] = hsv_mask
        blur_mask = cv2.medianBlur(hsv_masks[color], ksize=3)
        cv2.imshow("{}_mask_with_medianblur".format(color), hsv_masks[color])

        # ラベリング処理
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            hsv_masks[color]
        )
        print(labels)

        # 領域(stats[:, 4])が２つ以上ある場合(そのうち1つは背景)だけ処理
        if len(stats[:, 4]) >= 2:
            # 面積でソート、　今回は最も大きい領域１つだけ利用
            idx = stats[:, 4].argsort()[-2]

            # 領域の外接矩形の角座標を入手
            x0 = stats[idx, 0]
            y0 = stats[idx, 1]
            x1 = x0 + stats[idx, 2]
            y1 = y0 + stats[idx, 3]

            # 半径算出
            radius = (stats[idx, 2] + stats[idx, 3]) // 4

            # 円描画
            cv2.circle(
                frame,
                (int(centroids[idx, 0]), int(centroids[idx, 1])),
                radius,
                (0, 0, 255),
                5,
            )

            # 中心点描画
            cv2.circle(
                frame,
                (int(centroids[idx, 0]), int(centroids[idx, 1])),
                10,
                (168, 87, 167),
                -1,
            )

            # 領域の重心座標、サイズを表示 (引数 : 描画画像、 書き込む文字列、 書き込む座標、 フォント、 サイズ、 色、 太さ)
            cv2.putText(
                frame,
                "Center X: " + str(int(centroids[idx, 0])),
                (x1 - 30, y1 + 15),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "Center Y: " + str(int(centroids[idx, 1])),
                (x1 - 30, y1 + 30),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "Radius: " + str(radius),
                (x1 - 30, y1 + 45),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (0, 255, 255),
                2,
            )

    # 結果画像の表示
    cv2.imshow("labeling", frame)

    # 終了オプション
    k = cv2.waitKey(1)
    if k == ord("q"):
        break


# カメラリリース、windowの開放
cap.release()
cv2.destroyAllWindows()
