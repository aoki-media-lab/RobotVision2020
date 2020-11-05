# ライブラリのインポート
import cv2
import numpy as np

# ノイズ処理用のカーネルを定義
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# 今回はサンプル動画(.avi形式)を用いて背景差分を抽出
cap = cv2.VideoCapture("./image_data/opticalflow.avi")

# 背景差分のクラスを定義
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

while True:

    # 動画のフレームを取得
    ret, frame = cap.read()

    # 動画のフレームが無くなったら強制終了
    if not ret:
        break

    # 動画の表示
    cv2.imshow("video", frame)

    # 背景差分の計算→マスク出力　をしている部分
    # 毎フレーム fgpg.apply していくことで変化のない部分を背景と学習し、学習した背景との差分をマスク(fgmask)として出力
    fgmask = fgbg.apply(frame)

    # ノイズの除去処理
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    # 背景差分出力
    cv2.imshow("frame", fgmask)

    # 終了オプション
    k = cv2.waitKey(1)
    if k == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
