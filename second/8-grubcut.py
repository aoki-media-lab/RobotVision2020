# ライブラリのインポート
import copy

import cv2
import numpy as np

# アプリ用のスタジアム、ボール画像を読みこみ
ball_img = cv2.imread("./image_data/ball.png")
stadium_img = cv2.imread("./image_data/stadium.png")

# 適当な大きさにスタジアムをリサイズ
stadium_img = cv2.resize(stadium_img, (1200, 700))

# ボールの高さ、幅
ball_h, ball_w = ball_img.shape[0], ball_img.shape[1]

# ボールの中心座標をスタジアムの中心に設定
idx_h = stadium_img.shape[0] // 2
idx_w = stadium_img.shape[1] // 2

# ウィンドウに結果を表示
cv2.imshow("ball", ball_img)
cv2.imshow("stadium", stadium_img)

"""
grubcutの準備・実行
"""
# 前景部分を指定してあげる(引数 : x座標, y座標, 幅, 高さ)
# 今回はボール画像の全体が前景部分にあたるので、全体を指定
# x,y座標は0を指定するとエラーとなるため1を指定
cut_rect = (1, 1, ball_w, ball_h)

# grubcutに必要なmaskや座標を格納するための配列の準備
ball_mask = np.zeros((ball_h, ball_w), np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)


# grubcutの実行
cv2.grabCut(ball_img, ball_mask, cut_rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

"""
[grubcutの実行結果について]
ball_maskでは各画素が0~3で分類される。
それぞれ...
0.背景
1.前景
2.背景らしい
3.前景らしい
である。
そこで、0,2の画素は0として、1,3の画素は1とするようなmaskを用意してあげる
"""

# maskを用意
mask = np.where((ball_mask == 2) | (ball_mask == 0), 0, 1).astype("uint8")

# maskを元に、ボールについて表示する部分と表示しない（スタジアムのままの）部分を反映させる
# newball_img は合成画像用の変数
newball_img = np.zeros((ball_h, ball_w, 3), dtype="uint8")
for y in range(0, ball_h):
    for x in range(0, ball_w):
        if mask[y][x] == 0:  # 背景部分
            # (注意!)今回ボールの大きさが H:198、W:200と両方偶数のためこれで良いが、奇数の場合は工夫が必要
            newball_img[y][x] = stadium_img[idx_h - (ball_h // 2) + y][
                idx_w - (ball_w // 2) + x
            ]  # スタジアムを出力
        elif mask[y][x] == 1:  # 前景部分
            newball_img[y][x] = ball_img[y][x]  # ボールを出力

# ボールをスタジアムに配置
with_grubcut = copy.deepcopy(stadium_img)

# (注意!)今回ボールの大きさが H:198、W:200と両方偶数のためこれで良いが、奇数の場合は工夫が必要
with_grubcut[
    (idx_h - ball_h // 2) : (idx_h + ball_h // 2),
    (idx_w - ball_w // 2) : (idx_w + ball_w // 2),
] = newball_img


# 比較用にGrubcutせずにボールをスタジアムに配置
without_grubcut = copy.deepcopy(stadium_img)
without_grubcut[
    (idx_h - ball_h // 2) : (idx_h + ball_h // 2),
    (idx_w - ball_w // 2) : (idx_w + ball_w // 2),
] = ball_img


# ウィンドウに結果を表示
cv2.imshow("without_grubcut", without_grubcut)
cv2.imshow("with_grubcut", with_grubcut)

cv2.waitKey(0)
cv2.destroyAllWindows()
