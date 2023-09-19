import numpy as np
import matplotlib.pyplot as plt
import pickle


def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def load_cifar_10_data(data_dir, negatives=False):
    meta_data_dict = unpickle(data_dir + "/batches.meta")
    cifar_label_names = meta_data_dict[b'label_names']
    cifar_label_names = np.array(cifar_label_names)

    # training data
    cifar_train_data = None
    cifar_train_filenames = []
    cifar_train_labels = []

    for i in range(1, 5):
        cifar_train_data_dict = unpickle(data_dir + "/train_batch_{}".format(i))
        cifar_train_data = cifar_train_data_dict[b'data'] if i == 1 else np.vstack((cifar_train_data, cifar_train_data_dict[b'data']))
        cifar_train_filenames += cifar_train_data_dict[b'filenames']
        cifar_train_labels += cifar_train_data_dict[b'labels']

    cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3, 32, 32))
    cifar_train_filenames = np.array(cifar_train_filenames)
    cifar_train_labels = np.array(cifar_train_labels)

    # validation data
    cifar_val_data_dict = unpickle(data_dir + "/val_batch")
    cifar_val_data = cifar_val_data_dict[b'data']
    cifar_val_filenames = cifar_val_data_dict[b'filenames']
    cifar_val_labels = cifar_val_data_dict[b'labels']

    cifar_val_data = cifar_val_data.reshape((len(cifar_val_data), 3, 32, 32))
    cifar_val_filenames = np.array(cifar_val_filenames)
    cifar_val_labels = np.array(cifar_val_labels)

    # test data
    cifar_test_data_dict = unpickle(data_dir + "/test_batch")
    cifar_test_data = cifar_test_data_dict[b'data']
    cifar_test_filenames = cifar_test_data_dict[b'filenames']
    cifar_test_labels = cifar_test_data_dict[b'labels']

    cifar_test_data = cifar_test_data.reshape((len(cifar_test_data), 3, 32, 32))
    cifar_test_filenames = np.array(cifar_test_filenames)
    cifar_test_labels = np.array(cifar_test_labels)

    return cifar_train_data, cifar_train_filenames, cifar_train_labels, \
        cifar_val_data, cifar_val_filenames, cifar_val_labels, \
        cifar_test_data, cifar_test_filenames, cifar_test_labels, \
        cifar_label_names


if __name__ == "__main__":
    cifar_10_dir = 'cifar-10'

    train_data, train_filenames, train_labels, \
    val_data, val_filenames, val_labels, \
    test_data, test_filenames, test_labels, \
    label_names = \
        load_cifar_10_data(cifar_10_dir)

    print("Training data: ", train_data.shape)
    print("Training filenames: ", train_filenames.shape)
    print("Training labels: ", train_labels.shape)
    print("Validation data: ", val_data.shape)
    print("Validation filenames: ", val_filenames.shape)
    print("Validation labels: ", val_labels.shape)
    print("Test data: ", test_data.shape)
    print("Test filenames: ", test_filenames.shape)
    print("Test labels: ", test_labels.shape)
    print("Label names: ", label_names.shape)
