# (c) Shahar Gino, Feb-2017, sgino209@gmail.com

import numpy as np
import pandas as pd
from PIL import Image
from os import listdir
from sys import stdout
from sklearn import svm
from sklearn import datasets
from json import dump as json_dump
from scipy.ndimage import convolve
from matplotlib import pyplot as plt
from pickle import dump as pickle_dump
from skimage.filters import threshold_otsu
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split


labels = ["0","1","2","3","4","5","6","7","8","9",
          "A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
          "a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]

class Struct:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

# -----------------------------------------------------------------------------------------------------------
def import_and_clean_data(database_path, train_size, img_path):
    """
    Importing and Cleaning the Data:
    (-) Import raw image by filename (needs to be .Bmp)
    (-) Convert to Grayscale
    (-) Use Otsu's Thresholding method to reduce noise
    (-) Define "Nudging" to widen dataset and account for variance in signal location in image (used on training data)
    (-) Convert images to 1D np array of values
    """
    df = pd.read_csv(database_path, header=0)
    raw_y = np.asarray(df['Class'])
    raw_x = np.asarray([get_img(i, train_size, img_path) for i in df.index]).astype(float)
    x = np.asarray([i.ravel() for i in raw_x])
    y = raw_y
    return x, y

# -----------------------------------------------------------------------------------------------------------
def get_img_core(size, img_path):
    """
    Returns a binary image from my file directory with index i
    """
    img = Image.open(img_path)
    img = img.convert("L")
    img = img.resize((size, size))
    image = np.asarray(img)
    image.setflags(write=True)
    thresh = threshold_otsu(image.ravel())
    binary = image > thresh
    return binary

def get_img(i, size, img_base):
    return get_img_core(size, img_base + '/' + str(i + 1) + '.Bmp')

# -----------------------------------------------------------------------------------------------------------
def nudge_dataset(x, y, size):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the (size x size) images around by 1px to left, right, down, up
    """
    direction_vectors = [
        [[0, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [1, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0]]
    ]

    def shift(k, w):
        return convolve(k.reshape((size, size)), mode='constant', weights=w).ravel()

    xx = np.concatenate([x] + [np.apply_along_axis(shift, 1, x, vector) for vector in direction_vectors])
    yy = np.concatenate([y for _ in range(5)], axis=0)
    return xx, yy

# -----------------------------------------------------------------------------------------------------------
def train_core(name, classifier, train_img_base, img_size, pickle_en, pickle_path, print_cm_en, cm_dump_path):

    # Prepare for Training:
    x, y = import_and_clean_data(train_img_base + '/../trainLabels.csv', img_size, train_img_base)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    x_train, y_train = nudge_dataset(x_train, y_train, img_size)

    # Fit (=Train!):
    classifier.fit(x_train, y_train)

    # Analyze model quality (over Test partition):
    # (-) precision = TP / (TP + FP) --> low precision means high FP
    # (-) recall = TP / (TP + FN) --> low recall means high FN
    # (-) f1-score = 2 * ((precision * recall) / (precision + recall)) --> high F1 means low False detections
    # (-) support = the number of true instances for each label

    y_pred = classifier.predict(x_test)
    print name + ':\n' + str(classification_report(y_test, y_pred))
    print name + ':\n' + str(accuracy_score(y_test, y_pred))

    # Pickle (=save model):
    if pickle_en:
        with open(pickle_path + '/' + name + '.pkl', 'w') as picklefile:
            pickle_dump(classifier, picklefile)

    # Calculate and dump Confusion-Matrix:
    if print_cm_en:
        calc_and_print_confusion_matrix(name, y_test, y_pred, cm_dump_path)

    return classifier

# -----------------------------------------------------------------------------------------------------------
def train_svm_digits():

    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    classifier = svm.SVC(gamma=0.001)
    classifier.fit(data, digits.target)

    return classifier

# -----------------------------------------------------------------------------------------------------------
def calc_and_print_confusion_matrix(name, y_test, y_pred, cm_dump_path):

    cm = confusion_matrix(y_test, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    data = list(list(i) for i in cm)

    cm_data = {
        "columns": [list(["R", i]) for i in labels],
        "index": [list(i) for i in labels],
        "data": data,
    }

    cm_digits = {
        "columns": [list(["R", i]) for i in labels[:10]],
        "index": [list(i) for i in labels[:10]],
        "data": [i[:10] for i in data[:10]],
    }

    cm_caps = {
        "columns": [list(["R", i]) for i in labels[10:36]],
        "index": [list(i) for i in labels[10:36]],
        "data": [i[10:36] for i in data[10:36]],
    }

    cm_lower = {
        "columns": [list(["R", i]) for i in labels[36:]],
        "index": [list(i) for i in labels[36:]],
        "data": [i[36:] for i in data[36:]],
    }

    with open(cm_dump_path + '/' + name + '_cm_data.json', 'w') as outfile:
        json_dump(cm_data, outfile)

    with open(cm_dump_path + '/' + name + '_cm_digits.json', 'w') as outfile:
        json_dump(cm_digits, outfile)

    with open(cm_dump_path + '/' + name + '_cm_caps.json', 'w') as outfile:
        json_dump(cm_caps, outfile)

    with open(cm_dump_path + '/' + name + '_cm_lower.json', 'w') as outfile:
        json_dump(cm_lower, outfile)

# -----------------------------------------------------------------------------------------------------------
def classify_core(trained_classifier, test_img_base, img_size, plot_en):

    predictions = {}

    idx = 1
    for image_file in listdir(test_img_base):

        # Load image:
        test_image = get_img_core(img_size, test_img_base + '/' + image_file)

        # Predict:
        test_image_flat = test_image.astype('float64').ravel()
        prediction = str(trained_classifier.predict([test_image_flat])).strip('[]\'')
        predictions[image_file] = prediction

        # Plot (few first results):
        if plot_en and (idx <= 8):
            plt.subplot(2, 4, idx)
            plt.imshow(test_image, interpolation='nearest')
            plt.set_cmap('gray_r')
            plt.title('%s --> %s' % (image_file, prediction))

        stdout.write('.')
        stdout.flush()

        idx += 1

    if plot_en:
        plt.show()

    print('')

    return predictions

# -----------------------------------------------------------------------------------------------------------
def classify_svm_digits(trained_classifier, test_img_base, plot_en):

    predictions = {}

    idx = 1
    for image_file in listdir(test_img_base):

        # Classification:
        img = Image.open(test_img_base + '/' + image_file)
        img = img.convert("L")
        img = img.resize((8, 8))
        image = np.asarray(img, dtype=np.float64)
        image.setflags(write=True)
        thresh = threshold_otsu(image.ravel())
        binary = 255 * (image < thresh)
        predictions[image_file] = trained_classifier.predict(binary)

        # Result
        if plot_en and idx <= 8:
            plt.subplot(2, 4, idx)
            plt.imshow(binary, interpolation='nearest')
            plt.set_cmap('gray_r')
            plt.title('%s --> %i' % (image_file, predictions[image_file]))

        stdout.write('.')
        stdout.flush()

        idx += 1

    if plot_en:
        plt.show()

    print('')

    return predictions
