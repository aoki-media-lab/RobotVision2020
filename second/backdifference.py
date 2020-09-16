# ライブラリのインポート
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
screenshot = False

# 実行
while(True):

    ret, frame = cap.read()
    cv2.imshow('camera',frame)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if screenshot:
        fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
        fgmask = fgbg.apply(frame)
        fgmask = fgbg.apply(photo)
        cv2.imshow('flow',fgmask)

    # 終了オプション
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
    elif k == ord('s'):
        photo = frame
        screenshot = True


cap.release()
cv2.destroyAllWindows()