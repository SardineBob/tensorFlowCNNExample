import tensorflow as tf
from tensorflow.keras import datasets
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense
from keras.utils import np_utils
import tkinter as tk
from PIL import Image, ImageTk

labels = ["airplane", "automobile", "bird", "cat",
          "deer", "dog", "frog", "horse", "ship", "truck"]
# 取得訓練樣本、訓練標籤以及測試樣本、測試標籤(正確答案)
(trainImg, trainLab), (testImg, testLab) = datasets.cifar10.load_data()
# Normalize pixel values to be between 0 and 1(將RGB數值正規畫成0到1，可提升準確度)
trainImg = trainImg.astype("float32") / 255
testImg = testImg.astype("float32") /255
# 將Label(可視為答案)進行Onehot excoding轉換(轉換成0跟1陣列，假設共有10種答案，答案若是6，則只有第6個位置是1，其他都是0)
trainLab = np_utils.to_categorical(trainLab)
testLab = np_utils.to_categorical(testLab)
#####擷取影像特徵#####
# 建立線性堆疊模型容器
model = Sequential()
# 一個卷積計算，包含一個卷積層與一個池化層
# 加入第一個卷積層
model.add(
    Conv2D(
        filters=32,  # 隨機產生32個filter
        kernel_size=(3, 3),  # 每一個filter size為3*3
        padding="same",  # 卷積運算中，產生的卷積影像大小不變
        input_shape=(32, 32, 3),  # 指定輸入的影像是32*32且為RGB的3個色度
        activation="relu"  # 設定為ReLU運作函式
    )
)
# 加入一個Dropout層，在訓練迭代中，隨機放棄神經元，避免overfitting，Dropout(0.25)表示隨機放棄25%的神經元
model.add(
    Dropout(rate=0.25)
)
# 加入第一個池化層，這層用意在縮減取樣數，把32*32縮減為16*16，但數量維持不變為32個樣本
model.add(
    MaxPooling2D(pool_size=(2, 2))
)
# 加入第二個卷積層
model.add(
    Conv2D(
        filters=64,  # 隨機產生64個filter
        kernel_size=(3, 3),  # 每一個filter size為3*3
        padding="same",  # 卷積運算中，產生的卷積影像大小不變
        activation="relu"  # 設定為ReLU運作函式
    )
)
# 加入一個Dropout層，在訓練迭代中，隨機放棄神經元，避免overfitting，Dropout(0.25)表示隨機放棄25%的神經元
model.add(
    Dropout(rate=0.25)
)
# 加入第二個池化層，這次是第二次縮減取樣，所以會把16*16縮減為8*8，但數量維持為64個樣本
model.add(
    MaxPooling2D(pool_size=(2, 2))
)

#####建立神經網路#####
# 加入平坦層
# 將池化過64個8*8影像影像轉為一層陣列，64*8*8=4096，會有4096個float數字，對應到了4096個神經元
model.add(
    Flatten()
)
# 加入一個Dropout層，在訓練迭代中，隨機放棄神經元，避免overfitting，Dropout(0.25)表示隨機放棄25%的神經元
model.add(
    Dropout(rate=0.25)
)
# 加入隱藏層
# 到了這層，神經元會變成1024個，並且加上非線性激活函式，避免訓練結果變成線性，這邊一樣會隨機放棄25%的神經元
model.add(
    Dense(1024, activation='relu')
)
# 加入一個Dropout層，在訓練迭代中，隨機放棄神經元，避免overfitting，Dropout(0.25)表示隨機放棄25%的神經元
model.add(
    Dropout(rate=0.25)
)
# 加入輸出層，最後的結果我們分成10個影像類別(10個神經元)，並使用激活函式softmax輸出結果，轉換成預測每個類別的機率
model.add(
    Dense(10, activation='softmax')
)
# 印出模型摘要
print(model.summary())

#####進行訓練#####
# 設定訓練方法
model.compile(
    loss="categorical_crossentropy", # 損失函式，通常會使用cross_entropy交叉諦，訓練效果佳
    optimizer="adam", # 優化方法，通常會使用adam最優化方法，可讓訓練更快收斂，提高準確率
    metrics=["accuracy"] # 評估模型方法使用accuracy準確率
)
# 開始訓練
result = model.fit(
    trainImg, # 經過標準化處理的影像特徵值
    trainLab, # 影像真實的值(我理解成正確的分類)，這個經過One-hot encoding處理
    validation_split=0.2, # 驗證資料的比例，例如50000筆資料，會拿50000*0.8=40000筆資料訓練，10000筆資料驗證
    epochs=10, # 執行10次訓練週期
    batch_size=128, # 每批128筆資料
    verbose=1 # 顯示訓練過程
)
# 儲存訓練結果
model.save_weights("classModel.h3")
# 檢視訓練結果
print(result)
# 執行預測
testResult = model.predict_classes(testImg)
print("真實答案(前20筆):")
print(testLab[:20])
print("預測答案(前20筆):")
print(testResult[:20])

#win = tk.Tk()
#img = Image.fromarray(testImg[0])
#photo = ImageTk.PhotoImage(image=img)
#box = tk.Label(win, image=photo)
# box.pack()
#box.img = photo
#lab = tk.Label(win, text=labels[testLab[0][0]])
# lab.pack()

#print(trainImg[0], testImg[0])


# win.mainloop()
