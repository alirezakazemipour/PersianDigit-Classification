# Imports
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPool2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import os
from matplotlib import pyplot as plt
import numpy as np
import cv2

train_dir = './digit_dataset/train/'
train_labels = []
train_images_list = []

files = os.listdir(train_dir)
files.sort()

for idx, file in enumerate(files):
    images_list = os.listdir(train_dir + file)
    images_list.sort()
    train_labels.extend([idx for _ in range(len(os.listdir(train_dir + file)))])
    train_images_list.append(images_list)

train_data = []
indices = np.arange(len(train_images_list))

for idx, file in zip(indices, files):
    for addr in train_images_list[idx]:
        I = cv2.imread(os.path.join(train_dir + file, addr))
        # I = np.expand_dims(I, 2)
        train_data.append(I)

test_dir = './digit_dataset/test/'
files = os.listdir(test_dir)
files.sort()

test_labels = []
test_images_list = []

for idx, file in enumerate(files):
    images_list = os.listdir(test_dir + file)
    images_list.sort()
    test_labels.extend([idx for _ in range(len(os.listdir(test_dir + file)))])
    test_images_list.append(images_list)

test_data = []
indices = np.arange(len(test_images_list))

for idx, file in zip(indices, files):
    for addr in test_images_list[idx]:
        I = cv2.imread(os.path.join(test_dir + file, addr))
        # I = np.expand_dims(I, 2)
        test_data.append(I)

# Hyper parameters
IMAGE_SIZE = 32
NUM_CHANNELS = 3
NUM_CLASSES = 10

train_dir = "./train"
valid_dir = "./valid"
test_dir = "./test"

weights_name = "best_model.h5"

# Model construction

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPool2D(strides=2))
model.add(Conv2D(filters=48, kernel_size=(5, 5), padding='valid', activation='relu'))
model.add(MaxPool2D(strides=2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

# Model Compilation
opt = Adam(lr=5e-4)
model.compile(opt,
              loss="categorical_crossentropy",
              metrics=["acc"])

# Augment training and validation Dataset
x_train = np.array(train_data)
x_train = x_train / 255.0
y_train = to_categorical(train_labels)

x_val = np.array(test_data)
x_val = x_val / 255.0
y_val = to_categorical(test_labels)

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1)
datagen.fit(x_train)

checkpoint = ModelCheckpoint(filepath="weights.h5",
                             monitor="val_acc",
                             verbose=1,
                             save_best_only=True)

if os.path.exists("weights.h5"):
    model.load_weights("weights.h5")

else:
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=64), validation_data=(x_val, y_val),
                        steps_per_epoch=len(x_train) / 64,
                        epochs=30,
                        callbacks=[checkpoint])

# # Show history
# plt.figure(1, figsize=(15, 8))
#
# plt.subplot(1, 2, 1)
# plt.plot(model_history["loss"])
# plt.plot(model_history["val_loss"])
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Model loss")
# plt.legend(["train", "valid"])
# plt.show()
#
# plt.subplot(1, 2, 2)
# plt.plot(model_history["acc"])
# plt.plot(model_history["val_acc"])
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.title("Model accuracy")
# plt.legend(["train", "valid"])
# plt.show()
#
# # Augment test Dataset
# test_generator = dataGenerator.flow_from_directory(test_dir,
#                                                    target_size=(IMAGE_SIZE, IMAGE_SIZE),
#                                                    classes=None,
#                                                    batch_size=1,
#                                                    shuffle=False,
#                                                    seed=123)
#
# test_generator.reset()
#
# # Test and show the result
# pred = model.predict(test_generator, steps=len(test_generator), verbose=1)
#
# pred_class_indices = np.argmax(pred, axis=1)
#
# f, ax = plt.subplots(5, 5, figsize=(15, 15))
#
# for i in range(25):
#     bgr_img = cv2.imread(test_dir + test_generator.filenames[i])
#     rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
#
#     predicted_class = "Dog" if pred_class_indices[i] else "Cat"
#
#     ax[i // 5, i % 5].imshow(rgb_img)
#     ax[i // 5, i % 5].axis("off")
#     ax[i // 5, i % 5].set_title("Predicted:{}".format(predicted_class))
# plt.show()
