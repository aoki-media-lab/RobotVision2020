# ライブラリのインポート
import copy
import cv2
import numpy as np


# アプリ用のスタジアム、ボール画像を読みこみ
ball_img = cv2.imread("./image_data/ball.png")
stadium_img = cv2.imread("./image_data/stadium.png")

# スタジアムの大きさを適当に変更 (二つ目の引数は(w,h))
stadium_img = cv2.resize(stadium_img, (1200, 700))

# / ではなく // で切り捨て
# ボールの高さ、幅の[半分](半分だから注意！！)
# (注意!)今回ボールの大きさが H:198、W:200と両方偶数のためこれで良いが、奇数の場合は工夫が必要
ball_h, ball_w = ball_img.shape[0] // 2, ball_img.shape[1] // 2

# ボールの中心位置（中心座標)を初期ではスタジアムの中心に設定
idx_h = stadium_img.shape[0] // 2
idx_w = stadium_img.shape[1] // 2

print(idx_h)


while True:
    # スタジアムのコピーを作成
    stadium_copy = copy.deepcopy(stadium_img)

    # ボールの再配置
    stadium_copy[
        (idx_h - ball_h) : (idx_h + ball_h), (idx_w - ball_w) : (idx_w + ball_w)
    ] = ball_img

    # 結果画像の表示
    cv2.imshow("output", stadium_copy)

    # 終了オプション
    k = cv2.waitKey(1)
    if k == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
