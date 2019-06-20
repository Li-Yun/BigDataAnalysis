import numpy as np
import pandas as pd
from collections import defaultdict
from collections import Counter
import datetime
from numpy import linalg as LA
import os
import argparse

class Utility():
    def __init__(self, base_directory_path):
        self.base_path = base_directory_path

    def data_loading(self, file_name, flag):
        # read a csv file
        if flag == 1:  # including headers
            return pd.read_csv(file_name, skiprows = 1, header = None, low_memory=False)
        elif flag == 0:  # no headers
            return pd.read_csv(file_name, header = None, low_memory=False)

    def build_dic(self, in_array):
        res_dic = defaultdict()
        count = 0
        for index in range(in_array.shape[0]):
            if in_array[index, 0] not in res_dic:
                res_dic[in_array[index, 0]] = count
                count += 1
        return res_dic
    def mapping_string_to_num(self, in_dict, in_word):
        return in_dict[in_word]
    
    def average_bag_of_words_number(self, in_list):
        return sum(in_list) / len(in_list)
    
    def bag_of_words_extraction(self, str_list):
        return [list(Counter(single_str[0].split()).values()) for single_str in str_list]
    
    def date_to_weeks(self, single_date):
        today = datetime.date.today()
        single_date_list = single_date.split('-')
        specific_date = datetime.date(int(single_date_list[0]), int(single_date_list[1]), int(single_date_list[2]))
        return ((today - specific_date).days) / 7

    def data_preprocessing(self, data):
        
        new_data = np.zeros(data.shape)
        
        for index in range(data.shape[1]):
            if index == 2 or index == 3:
                new_data[:, index:index + 1] = data.iloc[:, index:index + 1].values
            elif index == 5:
                new_data[:, index:index + 1] = data.iloc[:, index:index + 1].values.astype(int)
            elif index == 6:
                mylen = np.vectorize(len)
                new_data[:, index:index + 1] = mylen(data.iloc[:, index:index + 1].values)
            elif index == 7:
                # build a dictionary
                word_dic = self.build_dic(data.iloc[:, index:index + 1].values)
                vect_mapping = np.vectorize(self.mapping_string_to_num)
                new_data[:, index:index + 1] = vect_mapping(word_dic, data.iloc[:, index:index + 1].values)
            elif index == 4:
                # compute bag-of-words features
                bag_features = self.bag_of_words_extraction(data.iloc[:, index:index + 1].values)
                # calculate average bag-of-words numbers
                vector_average_bag_of_words_number = np.vectorize(self.average_bag_of_words_number)
                new_data[:, index:index + 1] = vector_average_bag_of_words_number(bag_features).reshape(len(bag_features), 1)
            elif index == 1:
                # convert dates to the weeks
                vector_date_to_weeks = np.vectorize(self.date_to_weeks)
                new_data[:, index:index + 1] = vector_date_to_weeks(data.iloc[:, index:index + 1].values)
        return new_data[:, 1:]

    def save_predict_centroid_id(self, prediction, data, k_num):
        # create a directory
        folder_path = os.path.join(self.base_path, 'new_data')
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        # save multiple files based on the centroids
        data_ids = np.hstack((np.asarray(prediction).reshape(data.shape[0], 1), data))
        for k in range(k_num):
            pd.DataFrame(data_ids[:, [1, 2, 3, 4, 5, 6, 7, 8]][data_ids[:, 0] == k]).to_csv(
            os.path.join(folder_path, 'centroid_' + str(k) + '.csv'), header=None, index=None)

class K_means():
    def __init__(self, training_data, centroid_num = 3, tol_num = 1e-5, max_itr = 100):
        self.k = centroid_num
        self.tol_num = tol_num
        self.max_itr = max_itr
        self.train_data = training_data
        self.centroids = None
    
    def save_model(self):
        pd.DataFrame(self.centroids).to_csv('centroids.csv', header=None, index=None)
    
    def fit(self):
        # randomly choose a few data points as the centroids
        np.random.shuffle(self.train_data)
        self.centroids = self.train_data[:self.k, :]
        previous_centroids = self.centroids

        print('Training...')
        for epoch in range(self.max_itr):
            print('Epoch: ', epoch + 1)
            # compute the distance between centroids and points
            index_list = []
            for data_index in range(self.train_data.shape[0]):
                # for each point, find the cloest centroid
                diff_array = np.subtract(self.train_data[data_index, :], self.centroids)
                index_list.append( np.argmin(LA.norm(diff_array, axis=1)) )
            # calculate the number of points in each centroid
            centroid_count_list = list()
            for K_index in range(self.k):
                centroid_count_list.append(index_list.count(K_index))
            # update the centroids
            tmp_centroids = np.zeros(self.centroids.shape)
            for data_index in range(len(index_list)):
                tmp_centroids[index_list[data_index], :] += self.train_data[data_index, :]
            for K_index in range(self.k):
                tmp_centroids[K_index, :] = tmp_centroids[K_index, :] / centroid_count_list[K_index]
            self.centroids = tmp_centroids
            del tmp_centroids

            # stop-condition
            if (np.absolute(np.subtract(self.centroids, previous_centroids)) < self.tol_num).all():
                print('K-means achieves a stop condition at epoch', epoch + 1)
                break
            # update the previous centroids
            previous_centroids = self.centroids
        print('Training is complete !!')

    def predict(self, testing_data, model_path, utility):
        predict_res = []
        # load the model
        model = utility.data_loading(model_path, 0).values

        # iterate through all testing examples
        for data_index in range(testing_data.shape[0]):
            # for each point, find the cloest centroid
            diff_array = np.subtract(testing_data[data_index, :], model)
            predict_res.append( np.argmin(LA.norm(diff_array, axis=1)) )
        return predict_res

def main():
    # get arguments
    parser = argparse.ArgumentParser(description='Dig Data Clustering using K-means')
    parser.add_argument('--train', default = False, help = 'training flag')
    parser.add_argument('--predict', default = False, help = 'prediction flag')
    parser.add_argument('--train_ratio', default = 0.7, type=float, help = 'ratio of training data')
    parser.add_argument('--K', default = 7, type = int, help = 'centroid number')
    parser.add_argument('--max_itr', default = 2000, type = int, help = 'max iteration')
    parser.add_argument('--tol_num', default = 1e-5, type = float, help = 'stop threshold')
    arg = parser.parse_args()

    # create Utility class
    uitlity = Utility(os.getcwd())
    # data loading
    raw_data = uitlity.data_loading('Eluvio_DS_Challenge.csv', 1)
    data_set = uitlity.data_preprocessing(raw_data)
    np.random.seed(0)
    np.random.shuffle(data_set)
    # split the dataset to a train and test dataset
    training_num = int(data_set.shape[0] * arg.train_ratio)
    # create K-means class
    kmean = K_means(data_set[:training_num, :], centroid_num = arg.K, tol_num = arg.tol_num, max_itr = arg.max_itr)

    # train K-means
    if arg.train:
        kmean.fit()
        kmean.save_model()
    # test the testing data on the trained model
    if arg.predict:
        predict_centroid = kmean.predict(data_set[training_num:, :], 'centroids.csv', uitlity)
        # save centroid IDs
        uitlity.save_predict_centroid_id(predict_centroid, raw_data.values[training_num:, :], arg.K)

if __name__ == "__main__":
    main()
