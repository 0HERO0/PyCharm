import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#유리병 이미지를 리스트화 시킨다.
glass_images = list() #empty list
for i in range(60):
    file = "./glass/" + "image{0:02d}.jpg".format(i + 1)
    img = cv2.imread(file)
    if img is None:
        print("No File.")
        break
    img = cv2.resize(img,(64,64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    glass_images.append(img)

def plot_images(n_row:int, n_col:int, images:list[np.ndarray]) -> None:
    fig = plt.figure()
    (fig, ax) = plt.subplots(n_row, n_col, figsize = (n_col, n_row))
    for i in range(n_row):
        for j in range(n_col):
            axis = ax[i,j]
            axis.get_xaxis().set_visible(False)
            axis.get_yaxis().set_visible(False)
            axis.imshow(images[i*n_col + j])
    plt.show()
    return None

plot_images(n_row=6, n_col=10, images=glass_images)

#통조림 캔 이미지를 리스트화 시킨다.
can_images = list() #empty list
for i in range(60):
    file = "./can/" + "image{0:02d}.jpg".format(i + 1)
    img = cv2.imread(file)
    if img is None:
        print("No File.")
        break
    img = cv2.resize(img,(64,64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    can_images.append(img)

def plot_images(n_row:int, n_col:int, images:list[np.ndarray]) -> None:
    fig = plt.figure()
    (fig, ax) = plt.subplots(n_row, n_col, figsize = (n_col, n_row))
    for i in range(n_row):
        for j in range(n_col):
            axis = ax[i,j]
            axis.get_xaxis().set_visible(False)
            axis.get_yaxis().set_visible(False)
            axis.imshow(images[i*n_col + j])
    plt.show()
    return None

plot_images(n_row=6, n_col=10, images=can_images)

#플라스틱 병 이미지를 리스트화 시킨다.
plastic_images = list() #empty list
for i in range(60):
    file = "./plastic/" + "image{0:02d}.jpg".format(i + 1)
    img = cv2.imread(file)
    if img is None:
        print("No File.")
        break
    img = cv2.resize(img,(64,64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plastic_images.append(img)

def plot_images(n_row:int, n_col:int, images:list[np.ndarray]) -> None:
    fig = plt.figure()
    (fig, ax) = plt.subplots(n_row, n_col, figsize = (n_col, n_row))
    for i in range(n_row):
        for j in range(n_col):
            axis = ax[i,j]
            axis.get_xaxis().set_visible(False)
            axis.get_yaxis().set_visible(False)
            axis.imshow(images[i*n_col + j])
    plt.show()
    return None

plot_images(n_row=6, n_col=10, images=plastic_images)

# X_Train_data 만들기
X = glass_images + can_images + plastic_images
y = [[1,0,0]]*len(glass_images) + [[0,1,0]]*len(can_images) + [[0,0,1]]*len(plastic_images)
print(y)

### CNN 만들기
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(input_shape = (64,64,3),kernel_size=(3,3),filters = 32),
    tf.keras.layers.MaxPooling2D((2,2), strides=2),
    tf.keras.layers.Conv2D(kernel_size=(3,3),filters = 32),
    tf.keras.layers.MaxPooling2D((2,2), strides=2),
    tf.keras.layers.Conv2D(kernel_size=(3,3),filters = 32),
    tf.keras.layers.MaxPooling2D((2,2), strides=2),
    tf.keras.layers.Flatten(),
    # Nural Network
    tf.keras.layers.Dense(units=512, activation='relu'),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dense(units=3, activation='softmax'),
])
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
X = np.array(X)
y = np.array(y)
history = model.fit(x=X, y=y, epochs=1000)

#예시 리스트
example_images = list() #empty list
for i in range(15):
    file = "./example/" + "image{0:02d}.jpg".format(i + 1)
    img = cv2.imread(file)
    if img is None:
        print("No File.")
        break
    img = cv2.resize(img,(64,64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    example_images.append(img)

example_images = np.array(example_images)
plot_images(3,5, example_images)
predict_images = model.predict(example_images)
print(predict_images)

fig = plt.Figure()
(fig, ax) = plt.subplots(3,5,figsize=(10,4))
for i in range(3):
    for j in range(5):
        axis = ax[i,j]
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)
        if predict_images[i * 5 + j][2] > 0.7:
            axis.imshow(example_images[i*5+j])
plt.show()
