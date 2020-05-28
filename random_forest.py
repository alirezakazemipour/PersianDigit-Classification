import random
from sklearn.preprocessing import StandardScaler
from concurrent import futures
import cv2
from functools import partial
# Add to your imports
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os


def processing(dir, images_list, idx, file):
    print("Processing: " + file)
    temp = []

    for addr in images_list[idx]:
        I = cv2.imread(os.path.join(dir + file, addr))
        temp.append(I.ravel())

    return temp

# Don't bother yourself with this function
def extract_features(dir, images_list, files):
    data = []
    with futures.ProcessPoolExecutor() as executor:
        indices = np.arange(len(images_list))
        func = partial(processing, dir, images_list)
        results = executor.map(func, indices, files)

        for result in results:
            data.extend(result)
    return data


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

print("------------Feature extraction for train data set------------")
train_data = extract_features(dir=train_dir, images_list=train_images_list, files=files)
print("------------End of extraction--------------------------------")

scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)

classifier = RandomForestClassifier(n_estimators=200)
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

print("------------Feature extraction for test data set------------")
test_data = extract_features(dir=test_dir, images_list=test_images_list, files=files)
print("------------End of extraction-------------------------------")

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
