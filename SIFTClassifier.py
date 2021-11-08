import cv2
import numpy as np
import glob
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class SIFTImageClassifier:

    def __init__(self, n_features = 20, n_clusters = 100):
        self.training_path = "Data/Medical/Training/*"
        self.testing_path = "Data/Medical/Testing/*"
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.image_list = []
        self.dir_list = {}  # Class names
        self.features = []  # Feature vectors
        self.labels = []  # Label vectors
        self.matrix = []
        self.feature_clusters = []
        self.cluster_features = {}
        self.dim = (256,256)

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
                self.image_list.append(img_file)
                img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, self.dim, interpolation=cv2.INTER_AREA)

                # TODO: Extract features here and assign them to 'descriptors'
                # descriptors =

                for descriptor in descriptors:
                    self.features.append(descriptor)
                self.labels.append(dc)

    def siftFeatures(self, img):
        """This function extracts SIFT features for one image, as in img"""
        sift = cv2.xfeatures2d.SIFT_create(self.n_features)
        keypoints_sift, descriptors = sift.detectAndCompute(img, None)
        return descriptors

    def cluster(self):
        """This function clusters feature vectors from training data. Number of clusters can be determined when
        initialising an object. """
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)
        self.kmeans_ret = kmeans.fit(self.features)

    def labelFeatuersToClusters(self):
        """This function maps the training data features to cluster centers."""
        for i in range(len(self.features)):
            cluster = self.kmeans_ret.predict([self.features[i]]).item()
            if cluster in self.cluster_features:
                cluster_count = self.cluster_features[cluster]
                cluster_feature = {cluster: cluster_count + 1}
                self.cluster_features.update(cluster_feature)
            else:
                self.cluster_features[cluster] = 1
            self.feature_clusters.append(cluster)

    def normalise(self):
        """This function normalises feature vectors using StandardScaler."""
        self.scale = StandardScaler().fit(self.matrix)
        self.matrix = self.scale.transform(self.matrix)

    def createMatrix(self):
        """This function creates the visual word-image matrix to represent the set of training images."""
        image_feature_matrix = [[] for x in range(len(self.image_list))]  # rows represent images, columns represent features (visual words)
        for i in range(len(self.image_list)):
            image_features_list = image_feature_matrix[i]
            for j in range(self.n_clusters):
                image_features_list.append(0)
            for j in range(i * self.n_features, (i * self.n_features) + self.n_features):
                c = self.feature_clusters[j]
                cfreq = image_features_list[c]
                image_features_list[c] = cfreq + 1

        self.matrix = image_feature_matrix

    def train(self):
        """This function trains a support vector machine with features from training images, and their labels."""
        self.svm = SVC(C=1, kernel='sigmoid')
        self.svm.fit(self.matrix, self.labels)

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
                img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, self.dim, interpolation=cv2.INTER_AREA)

                # TODO: Extract features here and assign them to 'descriptors'
                # descriptors =

                feature_clusters = []
                cluster_features = {}

                for i in range(len(descriptors)):
                    cluster = self.kmeans_ret.predict([descriptors[i]]).item()
                    if cluster in cluster_features:
                        cluster_count = cluster_features[cluster]
                        cluster_feature = {cluster: cluster_count + 1}
                        cluster_features.update(cluster_feature)
                    else:
                        cluster_features[cluster] = 1
                    feature_clusters.append(cluster)

                image_features_list = []
                for j in range(self.n_clusters):
                    image_features_list.append(0)
                for j in range(0, self.n_features):
                    c = feature_clusters[j]
                    cfreq = image_features_list[c]
                    image_features_list[c] = cfreq + 1

                image_features_list = self.scale.transform([image_features_list])
                pred = self.svm.predict(image_features_list).tolist()[0]
                if (pred == self.dir_list[dir_name]):
                    correct_predictions += 1

            print('Class name: ' + dir_name)
            print('Correct Predictions: ' + str(correct_predictions) + ' out of ' + str(len(dir_images)))
        pass

#Main entry point:
if __name__ == '__main__':
    imclass = SIFTImageClassifier()
    imclass.extract()
    imclass.cluster()
    imclass.labelFeatuersToClusters()
    imclass.createMatrix()
    imclass.normalise()
    imclass.train()
    imclass.test()

