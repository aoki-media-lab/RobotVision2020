
# ライブラリのインポート
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# 実行
while True:

    ##  以下記述  ##

    # circle描画はURL参照→(http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_gui/py_drawing_functions/py_drawing_functions.html)


    # 終了オプション
    k = cv2.waitKey(1)
    if k == ord("q"):
        break


# カメラリリース、windowの開放
cap.release()
cv2.destroyAllWindows()