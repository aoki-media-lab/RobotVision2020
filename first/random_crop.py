import cv2

img_path = "./pakutaso_uyuni.jpg"

# 画像の読み込み
img = cv2.imread(img_path)

# 画像をランダムに切り取るコードをかく
# 画像は配列として読み込まれるので，
# 画像の切り取りは，配列の一部を取り出せばできる．
