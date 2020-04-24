from lenet import model
import os
import random
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from concurrent import futures
import numpy as np
from sklearn.svm import SVC
import cv2
from keras.models import Model
from matplotlib import pyplot as plt

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
model.layers.pop()
model.layers.pop()
model.layers.pop()
model.layers.pop()
# model.summary()
# print(model.layers[-1].output.shape)
model = Model(inputs=model.input, outputs=[model.layers[2].output])
# exit(0)
indices = np.arange(len(train_images_list))

for idx, file in zip(indices, files):
    for addr in train_images_list[idx]:
        I = cv2.imread(os.path.join(train_dir + file, addr))
        I = I / 255.0
        I = np.expand_dims(I, 0)
        F = model.predict(I)
        plt.figure()
        for i in range(1, 1 + 48):
            plt.subplot(6, 8, i)
            plt.imshow(F[0, :, :, i - 1], cmap="gray")
            plt.axis("off")
        plt.savefig("figure.png")
        plt.close("all")
        F = np.reshape(F, -1)
        train_data.append(F)

scaler = StandardScaler()
X = scaler.fit_transform(train_data)

C_range = np.logspace(-2, 6, 9)
gamma_range = np.logspace(-7, 1, 9)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

max_score = 0
C = 0
gamma = 0


def grid_search(gamma_range, C_range):
    print("Grid search started...")
    param_grid = dict(gamma=[gamma_range], C=[C_range])
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
    grid.fit(X, train_labels)
    return grid.best_params_, grid.best_score_


with futures.ProcessPoolExecutor() as executor:
    results = executor.map(grid_search, gamma_range, C_range)

    for result in results:
        print(f"Parameters are {result[0]} with a score of {result[1] * 100: 3.3f} %")
        if result[1] > max_score:
            max_score = result[1]
            C = result[0]['C']
            gamma = result[0]['gamma']

classifier = SVC(C=C, gamma=gamma)
classifier.fit(X, train_labels)

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
        I = I / 255.0
        I = np.expand_dims(I, 0)
        F = model.predict(I)
        F = np.reshape(F, -1)
        test_data.append(F)

print("------------Prediction on test data-------------")

idx = [random.randint(0, len(test_data) - 1) for i in range(10)]
test_input = [test_data[i] for i in idx]
test_labels = [test_labels[i] for i in idx]
test_input = scaler.fit_transform(test_input)
results = classifier.predict(test_input)
print('predictions: ', results)
print("train labels: ", list(set(train_labels)))
print("test labels: ", test_labels)
print("Accuracy: ", (np.sum(results == test_labels) / len(results)) * 100, "%")

print("------------Prediction on train data-------------")

idx = [random.randint(0, len(train_data) - 1) for i in range(10)]
test_input = [train_data[i] for i in idx]
test_labels = [train_labels[i] for i in idx]
test_input = scaler.fit_transform(test_input)
results = classifier.predict(test_input)
print('predictions: ', results)
print("train labels: ", list(set(train_labels)))
print("test labels: ", test_labels)
print("Accuracy: ", (np.sum(results == test_labels) / len(results)) * 100, "%")
