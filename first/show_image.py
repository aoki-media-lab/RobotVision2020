import cv2

img_path = "./pakutaso_uyuni.jpg"

# 画像の読み込み
img = cv2.imread(img_path)

# サイズの表示
print("画像のサイズ:", img.shape)

# 画像の表示
cv2.imshow("image", img)

# 0を押すと表示を終了
cv2.waitKey(0)
cv2.destroyAllWindows()
