"""
OpenCVを用いたオプティカルフロー
参考ページ : http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
"""

# ライブラリのインポート
import cv2
import numpy as np

# 今回はサンプル動画(.avi形式)を用いてオプティカルフローを観測
cap = cv2.VideoCapture("./image_data/opticalflow.avi")

"""
コーナー検出器のパラメータ設定
"""

# Shi-Tomasiのコーナー検出パラメータ
feature_params = dict(
    maxCorners=100,  # 保持するコーナー数, int
    qualityLevel=0.3,  # 最良値(最大固有値の割合?), float型
    minDistance=7,  # この距離内のコーナーを棄却, float型
    blockSize=7,
)  # 使用する近傍領域のサイズ, int

# Lucas-Kanade法のパラメータ
lk_params = dict(
    winSize=(15, 15),  # 検索ウィンドウのサイズ
    maxLevel=2,  # 検出器の構造を決める(そのままで良い)
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)  # 検索終了条件

# ランダムに色を１００個生成（値0～255の範囲で100行3列のランダムなndarrayを生成）
color = np.random.randint(0, 255, (100, 3))

"""
最初のフレームの処理
Webカメラを用いる場合は、キーボード操作とかで最初のフレームを指定しても良い
"""

# 最初のフレーム読み込み
first_flag, first = cap.read()

# グレースケールに変換
gray_first = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
feature_first = cv2.goodFeaturesToTrack(gray_first, mask=None, **feature_params)

# フロー書き出し用の画像作成
flow_mask = np.zeros_like(first)

"""
２枚目以降の処理
"""
while True:

    # 動画のフレーム取得
    ret, frame = cap.read()

    # 動画のフレームが無くなったら強制終了
    if not ret:
        break

    # グレースケールに変換
    gray_next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # オプティカルフロー検出
    # feature_next : gray_nextの特徴点の座標を保持
    feature_next, status, err = cv2.calcOpticalFlowPyrLK(
        gray_first, gray_next, feature_first, None, **lk_params
    )

    # オプティカルフローを検出した特徴点を選別（status → 0：検出せず、1：検出した）
    good_first = feature_first[status == 1]
    good_next = feature_next[status == 1]

    # オプティカルフローを描画
    for i, (next_point, first_point) in enumerate(zip(good_next, good_first)):

        # 前フレームの座標獲得
        first_x, first_y = first_point.ravel()

        # 後フレームの座標獲得
        next_x, next_y = next_point.ravel()

        # 前フレームと後フレームを繋ぐ線を描画
        flow_mask = cv2.line(
            flow_mask, (next_x, next_y), (first_x, first_y), color[i].tolist(), 2
        )

        # 現在の特徴点のところに丸（大きな点）を描画
        frame = cv2.circle(frame, (next_x, next_y), 5, color[i].tolist(), -1)

    output = cv2.add(frame, flow_mask)

    # ウィンドウに結果を表示
    cv2.imshow("window", output)

    # 終了オプション
    k = cv2.waitKey(1)
    if k == ord("q"):
        break

    # 次のフレーム、ポイントの準備
    gray_first = gray_next.copy()
    feature_first = good_next.reshape(-1, 1, 2)

# 終了処理
cv2.destroyAllWindows()
cap.release()
