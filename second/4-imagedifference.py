# ライブラリのインポート
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
# スクショしたかどうかを保存する変数 (まだ撮っていないのでFalse)
screenshot = False
# スクショを保存する変数
photo = None

# 実行
while True:

    # Webカメラのフレーム取得
    ret, frame = cap.read()
    cv2.imshow("camera", frame)

    # スクショがあるなら差分を出力
    if screenshot:
        # 背景差分のクラスを定義(リセット)       
        fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
        """
        3-backdifference.py では毎フレーム差分をとっていた(fgbg.apply)が、今回はphotoとframeの2回だけ。
        つまりphotoとframeの２フレームの直接的な差分をとっている。（画像間差分）
        """
        # 背景画像(スクショ)
        fgmask = fgbg.apply(photo)
        # 差分画像(カメラの入力フレーム)
        fgmask = fgbg.apply(frame)
        cv2.imshow("flow", fgmask)

    # 終了オプション
    k = cv2.waitKey(1)
    if k == ord("q"):
        break
    # フレームを保存 (スクショ)
    elif k == ord("s"):
        photo = frame
        screenshot = True

cap.release()
cv2.destroyAllWindows()
