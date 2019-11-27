# read data
# in major parts based on https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py
# licensed under Apache 2.0 by The TensorFlow Authors. All Rights Reserved.
# modified class Dataset for use without images, dropped some arguments

# in Anlehnung an https://github.com/healthDataScience/deep-learning-HAR  und
# https://stackoverflow.com/questions/40994583/how-to-implement-tensorflows-next-batch-for-own-data

# Copyright 2018 Thomas Hemmert-Pottmann
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from tensorflow.contrib.learn.python.learn.datasets import base  # for Datasets structure
import numpy as np  # misc
import pandas as pd  # for one-hot encoding
from sklearn.model_selection import train_test_split
import utilities.preprocessing as prep
from sklearn import preprocessing as skpreprocessing
import scipy.io
import copy
from skimage import transform
import joblib
import os
import random
import pickle
import io
import types
import argparse


class Dataset:
    """ class proto for Datasets """

    def __init__(self, data, labels):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._data = data
        self._labels = labels
        self._num_examples = data.shape[0]
        pass

    # @property
    # def data(self):
    #     return self._data
    #
    # @property
    # def labels(self):
    #     return self._labels

    def __getData(self):
        return self._data

    def __setData(self, x):
        self._data = x

    data = property(__getData, __setData)

    def __getLabels(self):
        return self._labels

    def __setLabels(self, x):
        self._labels = x

    labels = property(__getLabels, __setLabels)

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        """ returns a batch of size batch_size, shuffles data set every epoch (full iteration over complete data set)"""
        start = self._index_in_epoch
        # shuffle for first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            idx = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx)  # shuffle indexe
            self._data = self.data[idx]  # get list of `num` random samples
            self._labels = self.labels[idx]

        # go to the next batch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # get remainder in epoch
            rest_num_examples = self._num_examples - start
            data_rest_part = self.data[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            if shuffle:
                # shuffle data for next epoch
                idx0 = np.arange(0, self._num_examples)  # get all possible indexes
                np.random.shuffle(idx0)  # shuffle indexes
                self._data = self.data[idx0]  # get list of `num` random samples
                self._labels = self.labels[idx0]

            # start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples  # avoid the case where the #sample != integar times of batch_size
            end = self._index_in_epoch
            data_new_part = self._data[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate(
                (data_rest_part, data_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._data[start:end], self._labels[start:end]

    def add_to_dataset(self, data, labels=0):
        """adds data (rows) to the existing dataset"""
        if isinstance(data, pd.DataFrame):
            labels = data['Label_ISOLATION']
            data = data.drop('Label_ISOLATION', axis=1)
        self._data = np.concatenate((self._data, data), axis=0)
        self._labels = np.concatenate((self._labels, labels), axis=0)

    def combine_features(self, data, labels=0):
        if isinstance(data, pd.DataFrame):
            labels = data['Label_ISOLATION']
            data = data.drop('Label_ISOLATION', axis=1)
        elif isinstance(data, Dataset):
            labels = data.labels
            data = data.data
        if not np.array_equal(self.labels, labels):
            raise FeatureCombinationError('Labels are not consistent for feature combination')
        self._data = np.concatenate((self._data, data), axis=1)


class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class FeatureCombinationError(Error):
    """Exception raised for errors during feature combination in dataset class.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message


def read_file(name, path, file_format='csv'):
    """ read from comma-separated file or numpy file at given path
    :param name specify the file's name (include file extension)
    :param path specify the path to the file (include '/' at the end)
    :param file_format the format of the file which is loaded, e.g. 'csv', 'npy'
    :type name str
    :type path str
    :type file_format str
    :returns numpy array containing the data
    :rtype ndarray
    """
    if file_format == 'npy':
        data = np.load(path + name + '.npy')
    elif file_format == 'csv':
        data = np.genfromtxt(path + name + '.csv', delimiter=',')
    else:
        raise ValueError('Specified file format not available, check spelling')
    return data


def read_infotxt(datafolderpath):
    """read the Info.txt file written by Matlab
    :param datafolderpath
    :type datafolderpath str
    :returns the sequence length, number of sensors or channels, list of the sensors
    """
    path = datafolderpath + 'Info.txt'
    with open(path) as f:
        content = f.readlines()
    datapoints = int(content[2].split()[-1])
    sensors = content[4].split()
    n_sensors = len(sensors)
    return datapoints, n_sensors, sensors


def split_data(data, labels, ratio, shuffling=True):
    """ splits the given data+labels into two single datasets of type Dataset
    :param data: the dataset, formatted as array
    :param labels: corresponding labels for the dataset
    :param ratio: float between 0 and 1, corresponding percentage will be split for testing purposes
    :param shuffling: default True, shuffles the dataset before splitting
    :returns train and test dataset formatted as dataset
    :raises ValueError if ratio not between 0 and 1
    """
    if not 0 <= ratio <= 1:
        raise ValueError('ratio should be between 0 and 1.')
    # random_state = 42 ensures to generate the same random pattern for every run. test set will be the same every time
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=ratio,
                                                                        shuffle=shuffling, random_state=42)
    return make_dataset(data_train, labels_train), make_dataset(data_test, labels_test)


def make_one_hot(labels_int):
    """ One-hot encoding
    :param labels_int: integer encoded labels
    :returns one-hot encoded labels, labels used in the dataset as list, e.g. [0, 100, 101, 102]
    """
    # use pandas get_dummies because integers do not run from 0 to num_classes
    enc = pd.get_dummies(labels_int)  # creates dataframe
    labels_onehot, classlabels = enc.values, list(map(int, enc.columns))
    return labels_onehot, classlabels


def make_dataset(data, labels=0):
    """wrapper function for creating a Dataset"""
    if isinstance(data, pd.DataFrame) and "Label_ISOLATION" in data.columns:
        labels = data['Label_ISOLATION']
        data = data.drop('Label_ISOLATION', axis=1)
        dataset = Dataset(data=data, labels=labels)
    elif isinstance(data, list):
        for idx, element in enumerate(data):
            if isinstance(element, pd.DataFrame):
                if idx == 0:
                    dataset = make_dataset(element)
                else:
                    dataset.add_to_dataset(element)
    else:
        dataset = Dataset(data=data, labels=labels)
    return dataset


def make_datasets(train, val, test):
    """wrapper function for creating Datasets, containing Dataset train, validation and test"""
    return base.Datasets(train=train, validation=val, test=test)


def select_channels(data, avail_sensors, seq_length, sel_sensors):
    """
    :param data: the raw data read from .csv or .npy file
    :type data: ndarray
    :param avail_sensors: list with strings for all sensors contained in the data file
    :type avail_sensors: list
    :param seq_length: length of one data stream of a single sensor
    :type seq_length: int
    :param sel_sensors: list with strings specifying the sensors to be contained after reduction
    :type sel_sensors: list
    :return: reduced array with specified sensor signals, the reduced sensor list, new number of channels
    """
    new_sensor_count = len(sel_sensors)
    reduced_data = np.zeros([data.shape[0], seq_length * new_sensor_count])
    cnt = 0
    for sensor in sel_sensors:
        sel_sens_idx = avail_sensors.index(sensor)
        reduced_data[:, cnt * seq_length:(cnt + 1) * seq_length] = data[:, (seq_length * sel_sens_idx):(
                    seq_length * (sel_sens_idx + 1))]
        cnt += 1
    return reduced_data, sel_sensors, new_sensor_count


def do_preprocessing(data, preprocessing_type, n_channels, seq_length):
    img_dim = None
    if preprocessing_type == 'detrend':
        data = prep.detrend(data, n_channels, seq_length)
    elif preprocessing_type == 'standard':
        data = prep.detrend(data, n_channels, seq_length)
    elif preprocessing_type == 'zero2one':
        data = prep.detrend(data, n_channels, seq_length)
    elif preprocessing_type == 'neg_one2one':
        data = prep.detrend(data, n_channels, seq_length)
    elif preprocessing_type == 'gaussian':
        data = prep.detrend(data, n_channels, seq_length)
    elif preprocessing_type == 'fourier_samples':
        data = prep.detrend(data, n_channels, seq_length)
        data, seq_length = prep.fourier(data, n_channels, seq_length)
    elif preprocessing_type == 'fourier_channels':
        data = prep.detrend(data, n_channels, seq_length)
        data, seq_length = prep.fourier(data, n_channels, seq_length)
    elif preprocessing_type == 'greyscale':
        data = prep.detrend(data, n_channels, seq_length)
        data, img_dim = prep.greyscale_image(data, n_channels, seq_length)
        seq_length = img_dim[0] * img_dim[1]
    elif preprocessing_type == 'stft':
        data, img_dim = prep.stft(data, n_channels, seq_length)  # internally, linear detrend is applied
        seq_length = img_dim[0] * img_dim[1]
    elif preprocessing_type == 'wpi':
        data = prep.detrend(data, n_channels, seq_length)
        data, img_dim = prep.wpi(data, n_channels, seq_length)
        seq_length = img_dim[0] * img_dim[1]
    elif preprocessing_type == 'recurrence_plot':
        data = prep.detrend(data, n_channels, seq_length)
        data, img_dim = prep.recurrence_plot(data, n_channels, seq_length, downsample=True)
        seq_length = img_dim[0] * img_dim[1]
    elif preprocessing_type == 'gaf':
        data = prep.detrend(data, n_channels, seq_length)
        data, img_dim = prep.gasf_plot(data, n_channels, seq_length, downsample=True)
        seq_length = img_dim[0] * img_dim[1]
    return data, seq_length, img_dim
    

def handle_data(train_dir,
                sel_sensors=None,
                file_format='csv',
                preprocessing_type=None,
                split_ratio=0.2,
                one_hot=True):
    """ read in data sets, do preprocessing, do one hot encoding
    :param train_dir: string containing the path of the files
    :type train_dir: str
    :param sel_sensors: list with strings specifying the sensors to be contained after reduction
    :type sel_sensors: list
    :param file_format: string containing the format of the file which shall be loaded ('npy','csv')
    :type file_format: str
    :param preprocessing_type: specify the preprocessing step, e.g. 'standard', 'zero2one' etc.
    :type preprocessing_type: str
    :param split_ratio: specify how much data (in percentages) should be used for testing
    :type split_ratio: float
    :param one_hot: mode to choose if labels are returned as one_hot matrix (True) or numerical vector (False)
    :type one_hot: str
    :returns train and test dataset as Dataset, the classlabels, number of channels, sequence length, list of sensors
    :raises ValueError if labels in train and test set are not the same
    """

    # get data info
    seq_length, n_channels, sensors = read_infotxt(train_dir)

    # read the data
    try:
        print('loading dataset in npy format')
        data = read_file('dataset', train_dir, file_format='npy')
        labels = read_file('labels', train_dir, file_format='npy')
    except:
        print('npy format of data not found. Using csv')
        data = read_file('dataset', train_dir, file_format='csv')
        labels = read_file('labels', train_dir, file_format='csv')

    # reduce channels if specified
    if sel_sensors != sensors:
        data, sensors, n_channels = select_channels(data, sensors, seq_length, sel_sensors)

    # do preprocessing
    data, seq_length, img_dim = do_preprocessing(data, preprocessing_type, n_channels, seq_length)

    # shuffle and split into two datasets train and test with given ratio
    train, test = split_data(data, labels, split_ratio)

    # train scaler
    if preprocessing_type is 'fourier_samples':
        toscale = 'samples'
    else:
        toscale = 'channels'
    scaler = train_scaler(train.data, preprocessing_type, n_channels, seq_length, toscale=toscale)

    # transform data with scaler
    train.data = do_scaling(train.data, scaler, n_channels, seq_length, toscale=toscale)
    test.data = do_scaling(test.data, scaler, n_channels, seq_length, toscale=toscale)


    # do one-hot encoding
    train_labels, classlabels_train = make_one_hot(train.labels)
    test_labels, classlabels_test = make_one_hot(test.labels)

    # use one-hot encoding to generate numeric label vector with entries [0:labels-1]
    if not one_hot:
        train_labels = np.nonzero(train_labels)[1]
        test_labels = np.nonzero(test_labels)[1]

    # check if all labels are present in both sets
    if classlabels_test != classlabels_train:
        raise ValueError('Mismatch between labeled classes in training and test set: '
                         'train labels are %s and test labels are %s'
                         % (classlabels_train, classlabels_test))
    # return datasets with updated (one-hot encoded) labels
    return make_dataset(train.data, train_labels), make_dataset(test.data, test_labels), classlabels_train, \
           n_channels, seq_length, sensors, img_dim, scaler


def train_scaler(data, preprocessing_type, n_channels, seq_length, toscale='channels'):
    """scaling data with different modes, apply detrend per sample per channel
    :param data the dataset as array
    :type data ndarray
    :param n_channels the number of channels
    :type n_channels int
    :param seq_length the length of a sequence of one single sensor
    :type seq_length int
    :param preprocessing_type mode used for scaling, e.g. 'standard', 'zero2one'
    :type preprocessing_type str
    :param toscale mode used for scaling the time sequence of each channel('channels')
    or each timestep/feature over the corresponding samples('samples')
    :type toscale str
    :returns scaled dataset
    :rtype ndarray
    """

    scaler = None

    if toscale == 'channels':
        data = prep.concatenate_samples(data, n_channels, seq_length)
        data = data.transpose()

    if preprocessing_type == 'standard':
        # Center to the median and component wise scale according to the interquartile range
        scaler = skpreprocessing.RobustScaler().fit(data)

    elif preprocessing_type == 'gaussian':
        scaler = skpreprocessing.PowerTransformer().fit(data)   # scale to range [0,1]

    elif preprocessing_type == 'zero2one':
        scaler = skpreprocessing.MinMaxScaler().fit(data)   # scale to range [0,1]

    elif preprocessing_type == 'neg_one2one':
        scaler = skpreprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(data)  # scale to range [-1, 1]

    elif preprocessing_type == 'fourier_samples':
        scaler = skpreprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(data)  # scale to range [-1, 1]

    return scaler


def do_scaling(data, scaler, n_channels, seq_length, toscale='channels'):

    if scaler:

        if toscale == 'channels':
            data = prep.concatenate_samples(data, n_channels, seq_length)
            data = data.transpose()

        scaled = scaler.transform(data)

        if toscale == 'channels':
            scaled = scaled.transpose()
            scaled = prep.unconcatenate_samples(scaled, n_channels, seq_length)

    else:
        scaled = data

    return scaled


def load_scaler(folderpath_scaler=''):
    path = os.path.normpath(folderpath_scaler)
    path_seperated = path.split(os.sep)

    scaler = []
    while not scaler and len(path_seperated) >= 2:
        try:
            scaler = joblib.load(os.path.join(*path_seperated, 'scaler.save'))
            break
        except FileNotFoundError:
            print(os.path.join(*path_seperated, 'scaler.save') + ' not found')
            del path_seperated[-1]

    if not scaler:
        print('Scaler not found')
    else:
        print('loaded ' + os.path.join(*path_seperated, 'scaler.save'))
    return scaler

def seperate_classes(data, labels):
    """ seperate the data into classes
    :param data: data array with shape(n_examples x (num_channels*seqlength))
    :type data: ndarray
    :param labels: labels for the data
    :returns data_dict: data dictionary with n_label entries
    containing arrays with shape (n_corresponding_examples x (num_channels*seqlength))
    :raises ValueError: number of examples does not match between data and labels
    """
    if data.shape[0] == labels.shape[0]:
        data_dict = {}
        for i in range(max(labels) + 1):
            index = np.where(labels == i)[0]
            data_dict[i] = data[index, :]
    else:
        raise ValueError('Mismatch between number of examples in labels and data')
    return data_dict

def save_to_mat(variable, filename, variableName):
    tmpData = copy.deepcopy(variable)
    tmpData.columns = ['_'.join(col).strip() for col in tmpData.columns.values]
    scipy.io.savemat(filename, {variableName: tmpData.to_dict('list')}, long_field_names=True, oned_as='column')


def save_workspace(filename, var):
    # filename should contain the pkl ending
    # var should be the var = locals() from the base workspace
    # save_workspace('filename.pkl', locals())

    if not os.path.isdir(os.sep.join(os.path.split(filename)[:-1])):
        os.makedirs(os.sep.join(os.path.split(filename)[:-1]))

    varcopy = copy.copy(var)
    if 'var' in varcopy.keys():
        del varcopy['var']
    instancesToDelete = (types.ModuleType, types.FunctionType, io.IOBase, argparse.ArgumentParser)
    for key in var.keys():
        if isinstance(var[key], instancesToDelete):
            del varcopy[key]
    f = open(filename, 'wb')
    pickle.dump(varcopy, f)
    f.close()

    return 0


def load_workspace(filename):
    # filename should contain the pkl ending
    # unpack the data-variable in the base workspace as follows:
    # data = load_workspace('filename.pkl')
    # for key in data.keys():
    #     globals()[key] = data[key]
    #     del data

    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()

    return data


def initialize_multilevel_dict(key_list=None):

    result = dict()
    if len(key_list) >= 2:
        for key in key_list[0]:
            result[key] = initialize_multilevel_dict(key_list[1:])
    else:
        for key in key_list[0]:
            result[key] = dict()

    return result
