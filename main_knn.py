# (c) Shahar Gino, Feb-2017, sgino209@gmail.com

from common import *
from pickle import load as pickle_load
from sklearn.neighbors import KNeighborsClassifier
from knn_study import knn_study

# -----------------------------------------------------------------------------------------------------------
def main_knn(args):
    """
    K Nearest Neighbours:
    (-) https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm
    (-) http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    """

    # Generate a Trained classifier:
    if args.training_en:
        print('Generating a trained KNN classifier')
        trained_classifier = train_core('knn',
                                        KNeighborsClassifier(),
                                        args.train_img_base,
                                        args.img_size,
                                        args.pickle_en,
                                        args.pickle_path,
                                        args.print_cm_en,
                                        args.cm_dump_path)
    else:
        print('Loading a trained KNN classifier')
        with open(args.pickle_path + '/knn.pkl') as picklefile:
            trained_classifier = pickle_load(picklefile)

    # Classify test images:
    predictions = {}
    if args.classification_en:
        print('Classifying test images with the trained KNN classifier')
        predictions = classify_core(trained_classifier,
                                    args.test_img_base,
                                    args.img_size,
                                    args.plot_en)

    # KNN study:
    if args.knn_study_en:
        knn_study(range(1,10), args.train_img_base, args.img_size)

    return predictions
