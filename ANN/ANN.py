import os
import random
import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def load_data(data_dir):
    """Load dữ liệu (là test hay train phụ thuộc vào data_dir), trả về 2 danh sách:
    
    images: Mảng các ảnh.
    labels: Mảng các nhãn của anh.
    """
    # Mỗi thư mục con trong data_dir ứng với 1 nhãn. Nhãn là tên thư mục. 
    directories = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
    
    # Quét toàn bộ thư mục con để tạo 2 danh sách 
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f) 
                      for f in os.listdir(label_dir) if f.endswith(".ppm")]
        
    # Thêm ảnh và nhãn vào danh sách. 
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels


# Load datasets huấn luyện và dạy học. 
ROOT_PATH = "traffic"
train_data_dir = os.path.join(ROOT_PATH, "datasets/BelgiumTS/Training")
test_data_dir = os.path.join(ROOT_PATH, "datasets/BelgiumTS/Testing")

images, labels = load_data(train_data_dir)

# Set giúp 1 collection chứa các thành phần độc nhất, không trùng nhau. 
print("Unique Labels: {0}\nTotal Images: {1}".format(len(set(labels)), len(images)))

# Hiện nhãn và ảnh đại diện trực quan 
def display_images_and_labels(images, labels):
    unique_labels = set(labels)
    plt.figure(figsize=(15, 15))
    i = 1
    for label in unique_labels:
        # Chọn ảnh đại diện cho nhãn label.
        image = images[labels.index(label)]
        plt.subplot(8, 8, i)  # Tạo bảng đồ họa 8 x 8
        plt.axis('off')
        plt.title("Label {0} ({1})".format(label, labels.count(label)))
        i += 1
        _ = plt.imshow(image)

display_images_and_labels(images, labels)

# Hiện danh sách ảnh ứng với nhãn nào đó. Số lượng ảnh có giới hạn. 
def display_label_images(images, label):
    limit = 24  # Hiện tối đa 24 ảnh. 
    plt.figure(figsize=(15, 5))
    i = 1

    start = labels.index(label)
    end = start + labels.count(label)
    for image in images[start:end][:limit]:
        plt.subplot(3, 8, i)  
        plt.axis('off')
        i += 1
        plt.imshow(image)
    plt.show()

display_label_images(images, 32)

# Lấy 5 ảnh trong images, xuất kích thước. Min() và max() xuất ra giá trị một điểm ảnh lớn nhất hoặc nhỏ nhất. Lưu ý rằng nó dao động từ 0 -> 255, ta cần chuẩn hóa để giới hạn phạm vi giá trị này. 
for image in images[:5]:
    print("shape: {0}, min: {1}, max: {2}".format(image.shape, image.min(), image.max()))

plt.show()

# Tham chiếu preserve_range mặc định là false. skimage.transform.resize sẽ tự chuẩn hóa. 
images32 = [skimage.transform.resize(image, (32, 32), mode='constant')
                for image in images]
display_images_and_labels(images32, labels)

# Kiểm tra lại kết quả chuẩn hóa. 
for image in images32[:5]:
    print("shape: {0}, min: {1}, max: {2}".format(image.shape, image.min(), image.max()))

# Chuyển đổi list sang array. Các hàm API nhận array phổ biến hơn. 
labels_a = np.array(labels)
images_a = np.array(images32)
print("labels: ", labels_a.shape, "\nimages: ", images_a.shape)

# Graph là đối tượng chứa mô hình trong Tensorflow. 
graph = tf.Graph()

# Tạo mô hình trong graph. "with" trong Python tương tự using trong C#. 
with graph.as_default():
    # Placeholder đại diện cho dữ liệu đưa vào trong mô hình.
    images_ph = tf.placeholder(tf.float32, [None, 32, 32, 3])
    labels_ph = tf.placeholder(tf.int32, [None])

    # Lát phẳng (flat) ma trận: [None, height, width, channels]
    # sang 1 chiều: [None, height * width * channels] == [None, 3072]
    images_flat = tf.contrib.layers.flatten(images_ph)

    # Fully connected layer. 
    # Kích thước FC ứng với số nhãn (62). 
    logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)
    logits = tf.contrib.layers.fully_connected(logits, 62, tf.nn.relu)

    # Chuyển logits sang chỉ mục nhãn (có thể xem là nhãn khi nhãn cũng là sô chỉ mục).
    # Kích thước mặc định [None] => mảng 1 chiều có length == batch_size.
    predicted_labels = tf.argmax(logits, 1)

    # Định nghĩa hàm mất mát 
    # Dùng cross-entropy để phân loại.
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_ph))

    # Tạo đối tượng tối ưu bằng thuật toán Adam.
    train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    # Khởi tạo biến toàn cục trước khi chạy mô hình.
    init = tf.global_variables_initializer()

print("images_flat: ", images_flat)
print("logits: ", logits)
print("loss: ", loss)
print("predicted_labels: ", predicted_labels)

############# Huấn luyện 
# Session là lớp bao bọc các tiến trình trong graph khi chạy. 
session = tf.Session(graph=graph)

# Bước đầu tiên là khởi tạo các biến. 
_ = session.run([init])

for i in range(1000):
    _, loss_value = session.run([train, loss], 
                                feed_dict={images_ph: images_a, labels_ph: labels_a})
    if i % 10 == 0:
        print("Loss: ", loss_value)

############# Chạy mô hình để dự đoán 
# Chọn ngẫu nhiên 10 ảnh 
sample_indexes = random.sample(range(len(images32)), 10)
sample_images = [images32[i] for i in sample_indexes]
sample_labels = [labels[i] for i in sample_indexes]

# Chạy tiến trình "predicted_labels". "Tiến trình" dịch từ chữ "op", có thể dịch là phép tính toán. 
predicted = session.run([predicted_labels], 
                        feed_dict={images_ph: sample_images})[0]
print(sample_labels)
print(predicted)
# Hiện kết quả dự đoán trực quan. Debug để xem. 
fig = plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    prediction = predicted[i]
    plt.subplot(5, 2,1+i)
    plt.axis('off')
    color='green' if truth == prediction else 'red'
    plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction), 
             fontsize=12, color=color)
    plt.imshow(sample_images[i])


# Load dataset kiểm thử.
test_images, test_labels = load_data(test_data_dir)

# Tiền xử lý ảnh như phía trên.
test_images32 = [skimage.transform.resize(image, (32, 32), mode='constant')
                 for image in test_images]
display_images_and_labels(test_images32, test_labels)

# Chạy dự đoán toàn bộ ảnh test.
predicted = session.run([predicted_labels], 
                        feed_dict={images_ph: test_images32})[0]
# Tính toán accuracy.
match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])
accuracy = match_count / len(test_labels)
print("Accuracy: {:.3f}".format(accuracy))

session.close()
