import random
from sklearn.preprocessing import StandardScaler
from concurrent import futures
import cv2
from functools import partial
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from skimage import feature
import numpy as np
import os
import time

cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)


def processing(feature_type, dir, images_list, idx, file):
    print("Processing: " + file)
    temp = []

    for addr in images_list[idx]:
        I = cv2.imread(os.path.join(dir + file, addr))

        if feature_type == "raw_pixels":
            temp.append(I.ravel())

        elif feature_type == "hog":
            I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
            (H, _) = feature.hog(I, orientations=9, pixels_per_cell=(8, 8),
                                 cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1",
                                 visualize=True)
            temp.append(H)

        elif feature_type == "lbp":
            I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
            numPoints = 28
            radius = 3
            lbp = feature.local_binary_pattern(I, numPoints, radius)
            temp.append(lbp.ravel())

        elif feature_type == "hog_and_lbp":
            I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
            numPoints = 28
            radius = 3
            lbp = feature.local_binary_pattern(I, numPoints, radius)
            (H, _) = feature.hog(I, orientations=9, pixels_per_cell=(8, 8),
                                 cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1",
                                 visualize=True)
            temp.append(np.hstack([H, lbp.ravel()]))

    return temp


def extract_features(feature_type, dir, images_list, files):
    data = []
    with futures.ProcessPoolExecutor() as executor:
        indices = np.arange(len(images_list))
        func = partial(processing, feature_type, dir, images_list)
        results = executor.map(func, indices, files)

        for result in results:
            data.extend(result)
    return data


def grid_search(clf, train_data, train_labels, params):
    print(f"Grid search started for parameter:{params}...")
    grid = GridSearchCV(clf(), param_grid=params, cv=cv)
    grid.fit(train_data, train_labels)
    return grid.best_params_, grid.best_score_


def train(classifier_type, train_data, train_labels):
    max_score = 0
    if classifier_type == "linear_svm":
        C_range = np.logspace(-1, 6, 8)
        params = [dict(C=[C], kernel=["linear"]) for C in C_range]
        clf = SVC

    elif classifier_type == "rbf_svm":
        C_range = np.logspace(-1, 6, 8)
        gamma_range = np.logspace(-8, -1, 8)
        params = [dict(C=[C], gamma=[gamma]) for C, gamma in zip(C_range, gamma_range)]
        clf = SVC

    elif classifier_type == "knn":
        n_neighbors = np.arange(1, 11)
        params = [dict(n_neighbors=[n_neighbor]) for n_neighbor in n_neighbors]
        clf = KNeighborsClassifier

    elif classifier_type == "random_forest":
        n_estimators = np.arange(50, 201, step=10)
        params = [dict(n_estimators=[n_estimator]) for n_estimator in n_estimators]
        clf = RandomForestClassifier

    with futures.ProcessPoolExecutor() as executor:
        func = partial(grid_search, clf, train_data, train_labels)
        results = executor.map(func, params)

        for result in results:
            print(f"Parameters are {result[0]} with a score of {result[1] * 100: 3.3f} %")
            if result[1] > max_score:
                max_score = result[1]
                best_params = result[0]

    if classifier_type == "linear_svm":
        clf = SVC(C=best_params["C"], kernel=best_params["kernel"])

    elif classifier_type == "rbf_svm":
        clf = SVC(C=best_params["C"], gamma=best_params["gamma"])

    elif classifier_type == "knn":
        clf = KNeighborsClassifier(n_neighbors=best_params["n_neighbors"])

    elif classifier_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=best_params["n_estimators"])

    return clf.fit(train_data, train_labels)

classifier_type = "random_forest"  # "linear_svm", "rbf_svm", "knn", "random_forest
feature_type = "hog_and_lbp"  # "raw_pixels", "hog", "lbp", "hog_and_lbp"

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
train_data = extract_features(feature_type=feature_type, dir=train_dir, images_list=train_images_list, files=files)
print("------------End of extraction--------------------------------")

scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)

classifier = train(classifier_type, train_data, train_labels)

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
test_data = extract_features(feature_type=feature_type, dir=test_dir, images_list=test_images_list, files=files)
print("------------End of extraction-------------------------------")

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


