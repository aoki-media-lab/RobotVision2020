
# ライブラリのインポート
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# 実行
while(True):
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('camera',frame)
    # 終了オプション
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
