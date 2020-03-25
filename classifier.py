import numpy as np
import cv2
import os
import random
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from concurrent import futures

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


def train_processing(idx, file):
    print("Train Processing: " + file)
    temp = []

    for addr in train_images_list[idx]:
        I = cv2.imread(os.path.join(train_dir + file, addr))
        temp.append(I.ravel())
    return temp


with futures.ProcessPoolExecutor() as executor:
    indices = np.arange(len(train_images_list))
    results = executor.map(train_processing, indices, files)

    for result in results:
        train_data.extend(result)

scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)

cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
max_score = 0

n_estimators = np.arange(100, 200, step=20)
lrs = np.logspace(-1, 0, 5)[::-1]


def grid_search(n_estimators, lrs):
    print(f"Grid search started for parameter:{n_estimators, lrs}...")
    param_grid = dict(n_estimators=[n_estimators], learning_rate=[lrs])
    grid = GridSearchCV(AdaBoostClassifier(algorithm="SAMME.R"), param_grid=param_grid, cv=cv)
    grid.fit(train_data, train_labels)
    return grid.best_params_, grid.best_score_


with futures.ProcessPoolExecutor() as executor:
    results = executor.map(grid_search, n_estimators, lrs)

    for result in results:
        print(f"Parameters are {result[0]} with a score of {result[1] * 100: 3.3f} %")
        if result[1] > max_score:
            max_score = result[1]
            n_estimator = result[0]['n_estimators']
            lr = result[0]["learning_rate"]

classifier = AdaBoostClassifier(n_estimators=n_estimator, learning_rate=lr, algorithm="SAMME.R")
classifier.fit(train_data, train_labels)

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


def test_processing(idx, file):
    print("Test Processing: " + file)
    temp = []

    for addr in test_images_list[idx]:
        J = cv2.imread(os.path.join(test_dir + file, addr))
        temp.append(J.ravel())
    return temp


with futures.ProcessPoolExecutor() as executor:
    indices = np.arange(len(test_images_list))
    results = executor.map(test_processing, indices, files)

    for result in results:
        test_data.extend(result)

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
