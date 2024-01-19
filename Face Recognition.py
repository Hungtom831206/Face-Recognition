import tensorflow as tf
import numpy as np
import cv2
import os
from retinaface import RetinaFace
from skimage import transform as trans
import onnxruntime as ort
from sklearn.preprocessing import normalize

import sqlite3
import io
import time
import warnings
warnings.filterwarnings('ignore')
# 開啟視訊鏡頭
cap = cv2.VideoCapture(0)

# 等待1秒確保攝影機已經啟動
time.sleep(1)

# 讀取視訊鏡頭的畫面
ret, frame = cap.read()

# 儲存圖片
cv2.imwrite("captured_image.jpg", frame)

# 顯示圖片儲存成功訊息
print("圖片已成功儲存為 captured_image.jpg")

# 關閉視訊鏡頭
cap.release()



# init with normal accuracy option
detector = RetinaFace(quality='normal')
def face_detect(img_path, detector):
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    detections = detector.predict(img_rgb)

    return img_rgb, detections


# 這個是辨識影片為求方便會用到的
def face_detect_bgr(img_bgr, detector):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    detections = detector.predict(img_rgb)

    return img_rgb, detections



def face_align(img_rgb, landmarks):
    src = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041]], dtype=np.float32)

    dst = np.array(landmarks, dtype=np.float32).reshape(5, 2)

    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)

    M = tform.params[0:2, :]

    aligned = cv2.warpAffine(img_rgb, M, (112, 112), borderValue=0)

    return aligned

# 讀取圖檔，因為 opencv 預設是 bgr 因此要轉成 rgb
img_path = 'captured_image.jpg'
img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
detections = detector.predict(img_rgb)



onnx_path = 'model/arcface_r100_v1.onnx'
EP_list = ['CPUExecutionProvider']

# Create inference session
sess = ort.InferenceSession(onnx_path, providers=EP_list)

# 取得臉部位置 positions 及特徵點座標 landmark points
face_positions = []
face_landmarks = []
for i, face_info in enumerate(detections):
    face_positions = [face_info['x1'], face_info['y1'], face_info['x2'], face_info['y2']]
    face_landmarks = [face_info['left_eye'], face_info['right_eye'], face_info['nose'], face_info['left_lip'],
                      face_info['right_lip']]

# 取得對齊後的圖片並作轉置
aligned = face_align(img_rgb, face_landmarks)
t_aligned = np.transpose(aligned, (2, 0, 1))

# 將轉置後的人臉轉換 dtype 為 float32，並擴充矩陣維度，因為後續函式需要二維矩陣
inputs = t_aligned.astype(np.float32)
input_blob = np.expand_dims(inputs, axis=0)

# get the outputs metadata and inputs metadata
first_input_name = sess.get_inputs()[0].name
first_output_name = sess.get_outputs()[0].name

# inference run using image_data as the input to the model
# pass a tuple rather than a single numpy ndarray.
prediction = sess.run([first_output_name], {first_input_name: input_blob})[0]

# 進行正規化並且轉成一維陣列
final_embedding = normalize(prediction).flatten()
#from face_detection import face_detect
#from feature_extraction import feature_extract


def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)

    return sqlite3.Binary(out.read())


def convert_array(text):

    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


def load_file(file_path):
    file_data = {}
    for person_name in os.listdir(file_path):
        person_dir = os.path.join(file_path, person_name)

        person_pictures = []
        for picture in os.listdir(person_dir):
            picture_path = os.path.join(person_dir, picture)
            person_pictures.append(picture_path)

        file_data[person_name] = person_pictures

    return file_data

# 連接資料庫並取得內部所有資料
conn_db = sqlite3.connect('database.db')
cursor = conn_db.execute("SELECT * FROM face_info")
db_data = cursor.fetchall()
# 跟 database 中的數據做比較
total_distances = []
total_names = []
for data in db_data:
    total_names.append(data[1])
    db_embeddings = convert_array(data[2])
    distance = round(np.linalg.norm(db_embeddings - final_embedding), 2)
    total_distances.append(distance)

# 所有人比對的結果
total_result = dict(zip(total_names, total_distances))

# 找到距離最小者，也就是最像的人臉
idx_min = np.argmin(total_distances)

# 最小距離者的名字與距離
name, distance = total_names[idx_min], total_distances[idx_min]

# set threshold
threshold = 1.1

# 差異是否低於門檻
if distance < threshold:
    print('Found!', name, distance, total_result)
else:
    print('Unknown person', total_result)


img = cv2.imread('captured_image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # 將圖片轉成灰階

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")   # 載入人臉模型
faces = face_cascade.detectMultiScale(gray)    # 偵測人臉

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)    # 利用 for 迴圈，抓取每個人臉屬性，繪製方框
cv2.putText(img, name, (x,y-5),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
cv2.imshow('name', img)

cv2.waitKey(0) # 按下任意鍵停止
cv2.destroyAllWindows()