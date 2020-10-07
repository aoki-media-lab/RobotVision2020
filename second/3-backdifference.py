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
     
    #Webカメラのフレーム取得
    ret, frame = cap.read()
    cv2.imshow("camera", frame)

    # スクショがあるなら差分を出力
    if screenshot:
        fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
        fgmask = fgbg.apply(frame)
        fgmask = fgbg.apply(photo)
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
