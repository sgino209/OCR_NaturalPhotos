# (c) Shahar Gino, Feb-2017, sgino209@gmail.com

from common import *
from pickle import load as pickle_load
from sklearn import svm


# -----------------------------------------------------------------------------------------------------------
def main_svm(args):
    """
    Support Vectors Machines:
    (-) https://en.wikipedia.org/wiki/Support_vector_machine
    (-) http://scikit-learn.org/stable/modules/svm.html
    (-) http://scikit-learn.org/stable/auto_examples/datasets/plot_digits_last_image.html
    """

    # Generate a Trained classifier:
    if args.training_en:
        print('Generating a trained SVM classifier')
        if args.svm_digits_only:
            trained_classifier = train_svm_digits()

        else:
            trained_classifier = train_core('svm',
                                            svm.SVC(gamma=0.001),
                                            args.train_img_base,
                                            args.img_size,
                                            args.pickle_en,
                                            args.pickle_path,
                                            args.print_cm_en,
                                            args.cm_dump_path)

    else:
        print('Loading a trained SVM classifier')
        with open(args.pickle_path + '/svm.pkl') as picklefile:
            trained_classifier = pickle_load(picklefile)

    # Classify test images:
    predictions = {}
    if args.classification_en:
        print('Classifying test images with the trained SVM classifier')
        if args.svm_digits_only:
            predictions = classify_svm_digits(trained_classifier,
                                              args.test_img_base,
                                              args.plot_en)
        else:
            predictions = classify_core(trained_classifier,
                                        args.test_img_base,
                                        args.img_size,
                                        args.plot_en)

    return predictions
