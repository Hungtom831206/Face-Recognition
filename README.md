[人臉辨識.pdf](https://github.com/Hungtom831206/Face-Recognition/files/13997670/default.pdf)

人臉辨識:

(一)story:

As a 公司職員

I want 確保公司電腦不容易被非本公司員工登入 

So that 我想開發用臉部辨識登入電腦的程式

(二)方法:

1.RetinaFace

2.MTCNN

(三)步驟:

Step 1. Face Detection

偵測人臉並取得座標值

Step 2. Face Alignment

將人臉對齊，也就是將傾斜的人臉轉至端正的角度。

Step 3. Feature extraction

提取人臉特徵 (landmark points)，並進行特徵標準化 (Features Normalization)

Step 4. Create Database

創建資料庫並放入照片以供我們後續進行比對

Step 5. Face Recognition

將輸入的照片與資料庫中的照片進行比對，使用 L2-Norm(歐幾里得) 計算之間 最佳的距離 (distance)，可視為兩張人臉之 差異程度，給定threshold=1，

若 distance > threshold ⇒ 不同人臉，反之則視為同一張臉，比對照片找出最相似的人並判斷差異是否低於門檻



