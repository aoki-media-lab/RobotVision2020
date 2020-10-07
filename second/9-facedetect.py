# ライブラリのインポート
import cv2

"""
学習済のHaar-like特徴を用いた分類器のデータを利用
顔データ：haarcascade_frontalface_default.xml
瞳データ：haarcascade_eye.xml
このデータはOpenCVをインストールした時に入っている
今回はわかりやすいようにface_dataフォルダーに移動している。以下のリンクからも取得できる。
https://github.com/opencv/opencv/tree/master/data/haarcascades
[※補足]
今回のコードでは顔と瞳のみを検出しているが、笑顔(haarcascade_smile.xml)と上半身(haarcascade_upperbody.xml)
のデータも用意。是非試してみてください。
"""

# 学習済のHaar-like特徴を用いた分類器のデータをface_dataフォルダーから読み込み
face_cascade_path = "./face_data/haarcascade_frontalface_default.xml"
eye_cascade_path = "./face_data/haarcascade_eye.xml"

# 検出に用いるカスケード分類器の読み込み
face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()

    # グレースケールに変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 分類器で顔のx座標,y座標,幅,高さを取得
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # 各顔について
    for x, y, w, h in faces:

        # 顔の外接短形を描画
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = img[y : y + h, x : x + w]
        face_gray = gray[y : y + h, x : x + w]
        
        # 瞳検知・外接短形を描画
        eyes = eye_cascade.detectMultiScale(face_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(face, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # ウィンドウに結果を表示
    cv2.imshow("video image", img)

    # 終了オプション
    k = cv2.waitKey(1)
    if k == ord("q"):
        break

# 終了処理
cv2.destroyAllWindows()
cap.release()
