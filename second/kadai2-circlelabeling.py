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

# 実行
while True:
    # -----------以下記述-----------

    # circle描画はURL参照→(http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_gui/py_drawing_functions/py_drawing_functions.html)
    # circle描画の ”引数:線幅” を -1 に設定することで塗り潰しが可能(円の中心点の描画に必要!) 

    # 終了オプション
    k = cv2.waitKey(1)
    if k == ord("q"):
        break


# カメラリリース、windowの開放
cap.release()
cv2.destroyAllWindows()
