"""
Reference:
http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_core/py_image_arithmetics/py_image_arithmetics.html
"""
import cv2

cap = cv2.VideoCapture(0)

shizuku = cv2.imread("./shizuku.png")
H, W, C = shizuku.shape

# 組み込みたい画像のマスク画像を生成
img2gray = cv2.cvtColor(shizuku, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

# bitwise_andを用いて前景領域を抽出
shizuku_fg = cv2.bitwise_and(shizuku, shizuku, mask=mask)

# 実行
while True:
    # Webカメラのフレーム取得
    ret, frame = cap.read()
    cv2.imshow("camera", frame)

    # region of interest (注目領域)
    roi = frame[0:H, 0:W]

    # 注目領域内にある，埋め込みたい画像の部分を黒塗りする
    frame_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # 黒塗りした部分に，画像を埋め込む
    dst = cv2.add(frame_bg, shizuku_fg)
    frame[0:H, 0:W] = dst

    # キーボードの入力の受付
    k = cv2.waitKey(1)

    cv2.imshow("res", frame)

    # 終了
    if k == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
