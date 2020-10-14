import cv2
import numpy as np
from skimage.feature import hog
from sklearn.neighbors import NearestNeighbors

cap = cv2.VideoCapture(0)

# 特徴量の読み込み
features = np.load("./data/features.npy")
labels = np.load("./data/labels.npy")

# 最近傍探索のモデルを定義
model = NearestNeighbors(n_neighbors=1).fit(features)

# スクショしたかどうかを保存する変数 (まだ撮っていないのでFalse)
screenshot = False
# スクショを保存する変数
photo = None

# ノイズ除去のためのカーネルの定義
kernel = np.ones((5, 5), np.uint8)

# クラス名
class_name = "background"
LABEL2CLS = {0: "background", 1: "A", 2: "B"}

# 実行
while True:
    # Webカメラのフレーム取得
    ret, frame = cap.read()

    src = frame.copy()

    # 今映っている人のクラスを表示
    # putText(描画画像、 書き込む文字列、 書き込む座標、 フォント、 サイズ、 色、 太さ)
    cv2.putText(
        src, f"class: {class_name}", (30, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 1,
    )

    cv2.imshow("camera", src)

    # キーボードの入力の受付
    k = cv2.waitKey(1)

    # スクショがあるなら差分を出力
    if screenshot:
        fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
        fgmask = fgbg.apply(frame)
        fgmask = fgbg.apply(photo)
        cv2.imshow("flow", fgmask)

        # 白色領域のノイズを除去する
        fgmask = cv2.erode(fgmask, kernel)  # 収縮処理
        fgmask = cv2.dilate(fgmask, kernel)  # 膨張処理

        # マスクのかかっていない部分のみ切り取る
        # np.where(条件式) で，条件を満たすインデックスを取り出すことができる
        H_arr, W_arr = np.where(fgmask == 255)

        # マスクのかかっていない部分が存在しない場合(背景と一緒の場合)はwhile文の最初に戻る
        if len(H_arr) == 0:
            class_name = "background"
            continue

        left = min(W_arr)
        right = max(W_arr)
        top = min(H_arr)
        bottom = max(H_arr)

        # grayscaleに変換
        gray = cv2.cvtColor(
            frame[top : bottom + 1, left : right + 1], cv2.COLOR_BGR2GRAY
        )

        # リサイズ (H, W) = (56, 56) ※要調整
        gray = cv2.resize(gray, (56, 56))

        # HOGによって特徴抽出
        feat = hog(gray)

        # 配列の形を変更 (56*56,) => (1, 56*56)
        # feat = feat.reshape(1, 56 * 56) と同じ
        feat = feat.reshape(1, -1)

        # 最近傍探索. 二重のリストで結果が返ってくる.
        distances, indices = model.kneighbors(feat)

        label = labels[indices[0][0]]
        class_name = LABEL2CLS[label]

    # 終了
    if k == ord("q"):
        break
    # フレームを保存 (スクショ)
    elif k == ord("s"):
        photo = frame
        screenshot = True

cap.release()
cv2.destroyAllWindows()
