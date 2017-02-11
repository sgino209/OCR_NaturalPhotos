# (c) Shahar Gino, Feb-2017, sgino209@gmail.com

from common import *
from pickle import load as pickle_load
from sklearn.ensemble import RandomForestClassifier


# -----------------------------------------------------------------------------------------------------------
def main_rfc(args):
    """
    Random Forest Classfier:
    (-) https://en.wikipedia.org/wiki/Random_forest
    (-) http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    """

    # Generate a Trained classifier:
    if args.training_en:
        print('Generating a trained RFC classifier')
        trained_classifier = train_core('rfc',
                                        RandomForestClassifier(),
                                        args.train_img_base,
                                        args.img_size,
                                        args.pickle_en,
                                        args.pickle_path,
                                        args.print_cm_en,
                                        args.cm_dump_path)
    else:
        print('Loading a trained RFC classifier')
        with open(args.pickle_path + '/rfc.pkl') as picklefile:
            trained_classifier = pickle_load(picklefile)

    # Classify test images:
    predictions = {}
    if args.classification_en:
        print('Classifying test images with the trained RFC classifier')
        predictions = classify_core(trained_classifier,
                                    args.test_img_base,
                                    args.img_size,
                                    args.plot_en)

    return predictions
