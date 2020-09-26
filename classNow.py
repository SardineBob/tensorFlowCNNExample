from tensorflow.keras import datasets
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense
from PIL import Image, ImageTk
import numpy as np

labels = ["airplane", "automobile", "bird", "cat",
          "deer", "dog", "frog", "horse", "ship", "truck"]
# 取得訓練樣本、訓練標籤以及測試樣本、測試標籤(正確答案)
#(trainImg, trainLab), (testImg, testLab) = datasets.cifar10.load_data()
# Normalize pixel values to be between 0 and 1(將RGB數值正規畫成0到1，可提升準確度)
#trainImg = trainImg.astype("float32") / 255
#testImg = testImg.astype("float32") /255
# 將Label(可視為答案)進行Onehot excoding轉換(轉換成0跟1陣列，假設共有10種答案，答案若是6，則只有第6個位置是1，其他都是0)
#trainLab = np_utils.to_categorical(trainLab)
#testLab = np_utils.to_categorical(testLab)
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

# 讀取訓練結果
model.load_weights("classModel.h3")

# 讀取圖片
#img = Image.open("cat.jpg").resize((32,32))
img = Image.open("frog.jpg").resize((32,32))
byteImg = np.array(img)
byteImg = byteImg.astype("float32") / 255
listImg = np.array([byteImg])
print(listImg.shape)

# 執行預測
#testResult = model.predict_classes(testImg)
testResult = model.predict_classes(listImg)
print("真實答案(前20筆):")
#print(testLab[:20])
print("frog")
print("預測答案(前20筆):")
print(labels[testResult[0]])
