# (c) Shahar Gino, Feb-2017, sgino209@gmail.com

from common import *
from pickle import load as pickle_load
from sklearn.naive_bayes import BernoulliNB


# -----------------------------------------------------------------------------------------------------------
def main_bnb(args):
    """
    Bernoulli Naive Bayes:
    (-) https://en.wikipedia.org/wiki/Naive_Bayes_classifier
    (-) http://scikit-learn.org/stable/modules/naive_bayes.html#bernoulli-naive-bayes
    """

    # Generate a Trained classifier:
    if args.training_en:
        print('Generating a trained BNB classifier')
        trained_classifier = train_core('bnb',
                                        BernoulliNB(),
                                        args.train_img_base,
                                        args.img_size,
                                        args.pickle_en,
                                        args.pickle_path,
                                        args.print_cm_en,
                                        args.cm_dump_path)
    else:
        print('Loading a trained BNB classifier')
        with open(args.pickle_path + '/bnb.pkl') as picklefile:
            trained_classifier = pickle_load(picklefile)

    # Classify test images:
    predictions = {}
    if args.classification_en:
        print('Classifying test images with the trained BNB classifier')
        predictions = classify_core(trained_classifier,
                                    args.test_img_base,
                                    args.img_size,
                                    args.plot_en)

    return predictions
