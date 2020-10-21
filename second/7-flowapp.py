# ライブラリのインポート
import copy

import cv2
import numpy as np

# Webカメラ設定
cap = cv2.VideoCapture(0)

# Shi-Tomasiのコーナー検出パラメータ
feature_params = dict(
    maxCorners=100,  # 保持するコーナー数, int型
    qualityLevel=0.3,  # 最良値(最大固有値の割合?), float型
    minDistance=7,  # この距離内のコーナーを棄却, float型
    blockSize=7,
)  # 使用する近傍領域のサイズ, int

# Lucas-Kanade法のパラメータ
lk_params = dict(
    winSize=(15, 15),  # 検索ウィンドウのサイズ
    maxLevel=2,  # 検出器の構造を決める(そのままで良い)
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)  # 検索終了条件

# ランダムに色を１００個生成（値0～255の範囲で100行3列のランダムなndarrayを生成）
color = np.random.randint(0, 255, (100, 3))

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

# フレームカウント
count = 0

# 実行
while True:

    ret, frame = cap.read()

    # 10フレームに一回特徴点を更新
    if count % 10 == 0:
        # グレースケールに変換
        gray_first = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        feature_first = cv2.goodFeaturesToTrack(gray_first, mask=None, **feature_params)
        flow_mask = np.zeros_like(frame)  # フロー書き出し用の画像更新
        # カウントの初期化
        count = 0

    else:
        # グレースケールに変換
        gray_next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # オプティカルフロー検出
        feature_next, status, err = cv2.calcOpticalFlowPyrLK(
            gray_first, gray_next, feature_first, None, **lk_params
        )

        # オプティカルフローを検出した特徴点を選別（0：検出せず、1：検出した）
        good_first = feature_first[status == 1]
        good_next = feature_next[status == 1]

        # オプティカルフローを描画
        x_difs = 0
        y_difs = 0
        for i, (next_point, first_point) in enumerate(zip(good_next, good_first)):

            # 前フレームの座標獲得
            first_x, first_y = first_point.ravel()

            # 後フレームの座標獲得
            next_x, next_y = next_point.ravel()

            # 前後フレームのx, y方向の移動成分を加える
            x_difs += next_x - first_x
            y_difs += next_y - first_y

            # 前フレームと後フレームを繋ぐ線を描画
            flow_mask = cv2.line(
                flow_mask, (next_x, next_y), (first_x, first_y), color[i].tolist(), 2
            )

            # 現在の特徴点のところに丸（大きな点）を描画
            frame = cv2.circle(frame, (next_x, next_y), 5, color[i].tolist(), -1)

        output = cv2.add(frame, flow_mask)

        """
        オプティカルフローの移動ベクトルの向きに
        ボールを移動させる
        """
        # 全体のベクトル移動度が最大の方向を算出
        # ボールを動かすために必要なフロー変化量の大きさを閾値で設定、自分で実験して最適な値を探して！
        threshold = 100
        if np.abs(x_difs) < np.abs(y_difs) and np.abs(y_difs) >= threshold:
            if y_difs > 0:
                idx_h += 10
                print("↓")
            else:  # y_difs = 0 はありえないのでelseでまとめて良い
                idx_h -= 10
                print("↑")
        elif np.abs(x_difs) > np.abs(y_difs) and np.abs(x_difs) >= threshold:
            if x_difs > 0:
                idx_w += 10
                print("→")
            else:  # x_difs = 0 はありえないのでelseでまとめて良い
                idx_w -= 10
                print("←")
        else:
            print("・")

        """
        5-labelingapp.pyと同様のアプリ化
        ボールをスタジアムに配置する
        """
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

        # ウィンドウに結果を表示
        cv2.imshow("window", output)
        cv2.imshow("output", stadium)

        # 次のフレーム、ポイントの準備
        gray_first = gray_next.copy()
        feature_first = good_next.reshape(-1, 1, 2)

    # 終了オプション
    k = cv2.waitKey(1)
    if k == ord("q"):
        break

    # フレームカウント更新
    count += 1

# 終了処理
cv2.destroyAllWindows()
cap.release()
