import numpy as np
import pickle
from sliding_window import sliding_window
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
import torch
from scipy.io import loadmat


def opp_sliding_window(data_x, data_y, ws, ss):
    """
    Obtaining the windowed data from the HAR data
    :param data_x: sensory data
    :param data_y: labels
    :param ws: window size
    :param ss: stride
    :return: windows from the sensory data (based on window size and stride)
    """
    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))
    data_y = np.reshape(data_y, (len(data_y), ))  # Just making it a vector if it was a 2D matrix
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, ws, ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8)


# Data loader for the autoencoder
class HAR_Dataset(Dataset):
    def __init__(self, args, phase):
        """
        Defining the data loader
        :param args: arguments (from main.py)
        :param phase: train/val/test
        """
        # The .mat file containing the cleaned data
        self.filename = os.path.join(args.root_dir, args.data_file)

        # If the prepared dataset doesn't exist, give a message and exit
        if not os.path.isfile(self.filename):
            print('The preprocessed data is not available. Please run preprocess_data.py')
            exit(0)

        # Loading the data from the .mat file
        self.data_raw = self.load_dataset(self.filename)
        assert args.input_size == self.data_raw['train_data'].shape[1]

        # Obtaining the segmented data
        self.data, self.labels = opp_sliding_window(self.data_raw[phase + '_data'], self.data_raw[phase + '_labels'],
                                                    args.window, args.overlap)

        # Need to pick a percentage of the training dataset for analysis on dataset sizes required.
        # Have to randomize the frames before picking
        if phase == 'train':
            num_frames = self.data.shape[0]
            index = np.random.permutation(num_frames)
            fraction = int(np.floor(args.train_data_fraction * num_frames))
            frames_chosen = index[0:fraction]
            self.data = self.data[frames_chosen, :, :]
            self.labels = self.labels[frames_chosen]
            print('Size of the fraction of training data is {}, and the final sizes are {} and {}'.
                  format(args.train_data_fraction, self.data.shape, self.labels.shape))
            print('The unique labels present in the fraction are {}'.format(np.unique(self.labels)))

    def load_dataset(self, filename):
        """
        Loading the .mat file and creating a dictionary based on the phase
        :param filename: name of the .mat file
        :return: dictionary containing the sensory data
        """
        # Load the data from the .mat file
        data = loadmat(filename)

        # Putting together the data into a dictionary for easy retrieval
        data_raw = {'train_data': data['X_train'], 'train_labels': np.transpose(data['y_train']), 'val_data': data['X_valid'],
                    'val_labels': np.transpose(data['y_valid']), 'test_data': data['X_test'],
                    'test_labels': np.transpose(data['y_test'])}

        # Setting the variable types for the data and labels
        for set in ['train', 'val', 'test']:
            print('The shape of the {} dataset is {}, and the labels is {}'.format(set, data_raw[set+'_data'],
                                                                                   data_raw[set+'_labels']))
            data_raw[set + '_data'] = data_raw[set + '_data'].astype(np.float32)
            data_raw[set + '_labels'] = data_raw[set + '_labels'].astype(np.uint8)

        return data_raw

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Passes the windowed data based on the index
        :param index: index to choose the actual data point
        :return: data and lables for the training process
        """
        # Need to make the window 3 channels since it is being passed to the convolutional layer
        data = np.expand_dims(self.data[index, :, :], 0)
        data = torch.from_numpy(data).double()

        # Labels
        label = torch.from_numpy(np.asarray(self.labels[index])).double()
        return data, label


# Data loader for the MLP classifier
class MLP_Dataset(Dataset):
    def __init__(self, loc, phase):
        """
        Setting up the data loader for the MLP classifier. The inputs are the features learned from the autoencoder
        :param loc: location and file name of the stored features
        :param phase: train/val/test
        """
        # Loading the features
        self.all_data = pd.read_pickle(loc)

        self.data = self.all_data[phase+'_data']
        self.labels = self.all_data[phase + '_labels']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Passing the values to the training procedure
        :param index: index of the data loader
        :return: the learned features and corresponding labels
        """
        # Picking the data based on the index
        data = self.data[index, :]
        data = torch.from_numpy(data).double()

        # Corresponding label
        label = self.labels[index]
        label = torch.from_numpy(np.asarray(label)).double()

        return data, label


