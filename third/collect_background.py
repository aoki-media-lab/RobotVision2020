import glob

import cv2

cap = cv2.VideoCapture(0)

# スクショしたかどうかを保存する変数 (まだ撮っていないのでFalse)
screenshot = False
# スクショを保存する変数
photo = None

# フォルダにある画像の枚数を取得
data = glob.glob("./data/*.jpg")
n_data = len(data)

# 実行
while True:
    # Webカメラのフレーム取得
    ret, frame = cap.read()
    cv2.imshow("camera", frame)

    # キーボードの入力の受付
    k = cv2.waitKey(1)

    # 終了
    if k == ord("q"):
        break
    # 写真を保存
    elif k == ord("s"):
        cv2.imwrite(f"./data/{n_data}.jpg", frame)
        n_data += 1

cap.release()
cv2.destroyAllWindows()
