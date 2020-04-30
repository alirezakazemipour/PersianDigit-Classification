from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from skimage import feature
import numpy as np
from concurrent import futures
import cv2
import os
from functools import partial


class Classifier:
    def __init__(self, classifier="linear_svm", feature_type="hog_and_lbp"):
        self.clasifier = classifier
        self.feature_type = feature_type
        self.cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        self.numPoints = 28
        self.radius = 3

    def train(self, train_data, train_labels):

        max_score = 0

        n_estimators = np.arange(100, 200, step=20)
        lrs = np.logspace(-1, 0, 5)[::-1]

        with futures.ProcessPoolExecutor() as executor:
            results = executor.map(self.grid_search, n_estimators, lrs)

            for result in results:
                print(f"Parameters are {result[0]} with a score of {result[1] * 100: 3.3f} %")
                if result[1] > max_score:
                    max_score = result[1]
                    n_estimator = result[0]['n_estimators']
                    lr = result[0]["learning_rate"]

        classifier = AdaBoostClassifier(n_estimators=n_estimator, learning_rate=lr, algorithm="SAMME.R")
        classifier.fit(train_data, train_labels)

    def grid_search(self, n_estimators, lrs):
        print(f"Grid search started for parameter:{n_estimators, lrs}...")
        param_grid = dict(n_estimators=[n_estimators], learning_rate=[lrs])
        grid = GridSearchCV(AdaBoostClassifier(algorithm="SAMME.R"), param_grid=param_grid, cv=cv)
        grid.fit(train_data, train_labels)
        return grid.best_params_, grid.best_score_

    def test(self):
        pass

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
                lbp = feature.local_binary_pattern(I, self.numPoints, self.radius)
                temp.append(lbp.ravel())

            elif self.feature_type == "hog_and_lbp":
                I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
                lbp = feature.local_binary_pattern(I, self.numPoints, self.radius)
                (H, _) = feature.hog(I, orientations=9, pixels_per_cell=(8, 8),
                                     cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1",
                                     visualize=True)
                temp.append(np.hstack([H, lbp.ravel()]))

        return temp
