# Đây là file chính của chương trình để tạo ra model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
from lenet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import cv2
import os

# train_model.py -d Datasets/SMILEs

# Thiết lập tham số để thực thi bằng dòng lệnh
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Hãy chọn đường dẫn chứa dữ liệu")
args = vars(ap.parse_args())

# 1.Bước 1: Chuẩn bị dữ liệu
# 1.1 Thu thập dữ liệu: Đã thu thập lưu vào folder: datasets/SMILEs
# 1.2 Tiền xử lý
# - Khởi tạo danh sách data và labels
data = []
labels = []
# - Lặp qua folder datatet chứa ảnh đầu vào để nạp ảnh và lưu danh sách data
for image_path in sorted(list(paths.list_images(args["dataset"]))):
    # Nạp ảnh, tiền xử lý và lưu danh sách data
    image = cv2.imread(image_path) # Đọc ảnh
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Chuyển sang ảnh xám
    image = imutils.resize(image, width=28) # Thay đổi kích thước -> độ rộng là 28
# - Chuyển ảnh vào mảng
    image = img_to_array(image)
    data.append(image) # Nạp mảng (dữ liệu ảnh) vào danh sách data

# - Trích xuất class label từ đường dẫn ảnh và cập nhật danh sách labels
    label = image_path.split(os.path.sep)[-3] # -3 là do cấp folder là cấp 3
    label = "smilling" if label == "positives" else "not_smiling"
    labels.append(label) # Nạp nhãn vào danh sách labels

# - Covert mức xám pixel vào vùng [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# - Chuyển đổi labels từ biểu diễn số nguyên sang vectors
le = LabelEncoder().fit(labels)
labels = np_utils.to_categorical(le.transform(labels), 2)

# - Tách dữ liệu vào 02 nhóm: training data (80%) và testing data (20%)
(train_x, test_x, train_y, test_y) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

# 2. Bước 2: Tạo model (mạng); lớp Lenet là lớp chứa các lệnh tạo model, lớp. Lưu trong file lenet.py
print("[INFO]: Biên dịch model....")
model = LeNet.build(width=28, height=28, depth=1, classes=2)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# 3. Bước 3. Trainning mạng
print("[INFO]: Đang trainning model....")
H = model.fit(train_x, train_y, validation_data=(test_x, test_y), batch_size=32, epochs=15, verbose=1)
#H = model.fit(train_x, train_y, validation_data=(test_x, test_y), class_weight=class_weight, batch_size=64, epochs=15, verbose=1)

# 4. Bước 4. Đánh giá model (mạng)
print("[INFO]: Đánh giá model....")
predictions = model.predict(test_x, batch_size=64)
print(classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_))

# Lưu model vào đĩa
print("[INFO]: Lưu model network vào file....")
model.save("lenet.hdf5")
model.summary()

# Vẽ biểu đồ giá trị loss và độ chính xác khi training + testing
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 15), H.history["loss"], label="Mất mát khi trainning")
plt.plot(np.arange(0, 15), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 15), H.history["accuracy"], label="Độ chính xác khi trainning")
plt.plot(np.arange(0, 15), H.history["val_accuracy"], label="val_acc")
plt.title("Biểu đồ hiển thị mất mát trong Training và độ chính xác")
plt.xlabel("Epoch #")
plt.ylabel("Mất mát/Độ chính xác")
plt.legend()
plt.show()
