import glob

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# スクショしたかどうかを保存する変数 (まだ撮っていないのでFalse)
screenshot = False
# スクショを保存する変数
photo = None

# フォルダにある画像の枚数を取得
data = glob.glob("./data/*.jpg")
n_data = len(data)

# ノイズ除去のためのカーネルの定義
kernel = np.ones((5, 5), np.uint8)

# 実行
while True:
    # Webカメラのフレーム取得
    ret, frame = cap.read()
    cv2.imshow("camera", frame)

    # キーボードの入力の受付
    k = cv2.waitKey(1)

    # スクショがあるなら差分を出力
    if screenshot:
        fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
        fgmask = fgbg.apply(frame)
        fgmask = fgbg.apply(photo)
        cv2.imshow("flow", fgmask)

        if k == ord("c"):
            cv2.imwrite("./mask_results/initial_mask.jpg", fgmask)

            # 白色領域のノイズを除去する
            fgmask = cv2.erode(fgmask, kernel)  # 収縮処理
            cv2.imwrite("./mask_results/eroded_mask.jpg", fgmask)

            fgmask = cv2.dilate(fgmask, kernel)  # 膨張処理
            cv2.imwrite("./mask_results/denoised_mask.jpg", fgmask)

            # マスクのかかっていない部分のみ切り取る
            # np.where(条件式) で，条件を満たすインデックスを取り出すことができる
            H_arr, W_arr = np.where(fgmask == 255)

            left = min(W_arr)
            right = max(W_arr)
            top = min(H_arr)
            bottom = max(H_arr)

            # 不要な部分は無視して画像の保存
            cv2.imwrite(
                f"./data/{n_data}.jpg", frame[top : bottom + 1, left : right + 1]
            )

            n_data += 1

    # 終了
    if k == ord("q"):
        break
    # フレームを保存 (スクショ)
    elif k == ord("s"):
        photo = frame
        screenshot = True

cap.release()
cv2.destroyAllWindows()
