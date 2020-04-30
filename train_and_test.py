from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from skimage import feature
import numpy as np
from concurrent import futures
import cv2
import os
from functools import partial


class Classifier:
    def __init__(self, classifier_type="linear_svm", feature_type="hog_and_lbp"):
        self.classifier_type = classifier_type
        self.feature_type = feature_type
        self.cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        self.classifier = None

    def train(self, train_data, train_labels):

        max_score = 0
        if self.classifier_type == "linear_svm":
            C_range = np.logspace(-1, 6, 8)
            params = [dict(C=[C], kernel=["linear"]) for C in C_range]
            clf = SVC

        elif self.classifier_type == "rbf_svm":
            C_range = np.logspace(-1, 6, 8)
            gamma_range = np.logspace(-8, -1, 8)
            params = [dict(C=[C], gamma=[gamma]) for C, gamma in zip(C_range, gamma_range)]
            clf = SVC

        elif self.classifier_type == "knn":
            n_neighbors = np.arange(1, 11)
            params = [dict(n_neighbors=[n_neighbor]) for n_neighbor in n_neighbors]
            clf = KNeighborsClassifier

        elif self.classifier_type == "random_forest":
            n_estimators = np.arange(50, 201, step=10)
            params = [dict(n_estimators=[n_estimator]) for n_estimator in n_estimators]
            clf = RandomForestClassifier

        with futures.ProcessPoolExecutor() as executor:
            func = partial(self.grid_search, clf, train_data, train_labels)
            results = executor.map(func, params)

            for result in results:
                print(f"Parameters are {result[0]} with a score of {result[1] * 100: 3.3f} %")
                if result[1] > max_score:
                    max_score = result[1]
                    best_params = result[0]

            if self.classifier_type == "linear_svm":
                self.classifier = SVC(C=best_params["C"], kernel=best_params["kernel"])

            elif self.classifier_type == "rbf_svm":
                self.classifier = SVC(C=best_params["C"], gamma=best_params["gamma"])

            elif self.classifier_type == "knn":
                self.classifier = KNeighborsClassifier(n_neighbors=best_params["n_neighbors"])

            elif self.classifier_type == "random_forest":
                self.classifier = RandomForestClassifier(n_estimators=best_params["n_estimators"])

            self.classifier.fit(train_data, train_labels)

    def grid_search(self, clf, train_data, train_labels, params):
        print(f"Grid search started for parameter:{params}...")
        grid = GridSearchCV(clf(), param_grid=params, cv=self.cv)
        grid.fit(train_data, train_labels)
        return grid.best_params_, grid.best_score_

    def predict(self, x):
        return self.classifier.predict(x)

    def extract_features(self, dir, images_list, files):

        data = []
        with futures.ProcessPoolExecutor() as executor:
            indices = np.arange(len(images_list))
            func = partial(self.processing, dir, images_list)
            results = executor.map(func, indices, files)

            for result in results:
                data.extend(result)
        return data

    def processing(self, dir, images_list, idx, file):
        print("Processing: " + file)
        temp = []

        for addr in images_list[idx]:
            I = cv2.imread(os.path.join(dir + file, addr))

            if self.feature_type == "raw_pixels":
                temp.append(I.ravel())

            elif self.feature_type == "hog":
                I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
                (H, _) = feature.hog(I, orientations=9, pixels_per_cell=(8, 8),
                                     cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1",
                                     visualize=True)
                temp.append(H)

            elif self.feature_type == "lbp":
                I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
                numPoints = 28
                radius = 3
                lbp = feature.local_binary_pattern(I, numPoints, radius)
                temp.append(lbp.ravel())

            elif self.feature_type == "hog_and_lbp":
                I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
                numPoints = 28
                radius = 3
                lbp = feature.local_binary_pattern(I, numPoints, radius)
                (H, _) = feature.hog(I, orientations=9, pixels_per_cell=(8, 8),
                                     cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1",
                                     visualize=True)
                temp.append(np.hstack([H, lbp.ravel()]))

        return temp
