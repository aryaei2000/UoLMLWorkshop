import cv2
import numpy as np
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class HistogramImageClassifier:
    def __init__(self, n_features = 256):
        self.training_path = "Data/Fruits/Training/*" # Where training data is
        self.testing_path = "Data/Fruits/Testing/*" # Where testing data is
        self.n_features = n_features # Number of features in a histogram bin
        self.image_list = [] # List of training images
        self.dir_list = {}  # Class names
        self.features = []  # Feature vectors
        self.labels = []  # Label vectors
        self.dim = (256,256) # Resizing dimension

    def extract(self):
        """This function iterates through training images, extracts features from them, and saves feature vectors in
        self.features."""
        class_dirs = glob.glob(self.training_path)
        for dc in range(len(class_dirs)):
            dir_path = class_dirs[dc]
            dir_path_array = dir_path.split('/')
            dir_name = dir_path_array[len(dir_path_array) - 1]
            self.dir_list[dir_name] = dc

            dir_images = glob.glob(dir_path + '/*.*')

            for img_file in dir_images:
                img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, self.dim, interpolation=cv2.INTER_AREA)

                # TODO: Extract features here and assign them to 'descriptors'
                # descriptors =

                for descriptor in descriptors:
                    self.features.append(descriptor)
                self.labels.append(dc)

    def histogramFeatures(self, img):
        """This function extracts histogram features for one image, as in img"""
        histogram_vector = cv2.calcHist(img, [0], None, [self.n_features], [0, self.n_features])
        histogram_list = histogram_vector.tolist()
        feature_vector = []
        for numv in histogram_list:
            feature_vector.append(numv[0])
        return [feature_vector]

    def normalise(self):
        """This function normalises feature vectors using StandardScaler."""
        self.scale = StandardScaler().fit(self.features)
        self.features = self.scale.transform(self.features)

    def train(self):
        """This function trains a support vector machine with features from training images, and their labels."""
        self.svm = SVC(C=1, kernel='sigmoid')
        self.svm.fit(np.array(self.features),  np.array(self.labels))

    def test(self):
        """This function tests the trained model, and reports accuracy."""
        class_dirs = glob.glob(self.testing_path)
        for dc in range(len(class_dirs)):
            dir_path = class_dirs[dc]
            dir_path_array = dir_path.split('/')
            dir_name = dir_path_array[len(dir_path_array) - 1]
            self.dir_list[dir_name] = dc

            dir_images = glob.glob(dir_path + '/*.*')

            correct_predictions = 0
            for img_file in dir_images:
                self.image_list.append(img_file)
                img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, self.dim, interpolation=cv2.INTER_AREA)

                # TODO: Extract features here and assign them to 'image_features_list'
                # image_features_list =

                image_features_list = self.scale.transform(image_features_list)
                pred = self.svm.predict(image_features_list).tolist()[0]
                if (pred == self.dir_list[dir_name]):
                    correct_predictions += 1
            print('Class name: ' + dir_name)
            print('Correct Predictions: ' + str(correct_predictions) + ' out of ' + str(len(dir_images)))

#Main entry point:
if __name__ == '__main__':
    imclass = HistogramImageClassifier()
    imclass.extract()
    imclass.normalise()
    imclass.train()
    imclass.test()




