#   ___   ____ ____    _   _       _                   _   ____  _           _               A generic Machine-Learning
#  / _ \ / ___|  _ \  | \ | | __ _| |_ _   _ _ __ __ _| | |  _ \| |__   ___ | |_ ___  ___    environment, based on
# | | | | |   | |_) | |  \| |/ _` | __| | | | '__/ _` | | | |_) | '_ \ / _ \| __/ _ \/ __|   python sklearn toolkit,
# | |_| | |___|  _ <  | |\  | (_| | |_| |_| | | | (_| | | |  __/| | | | (_) | |_ (_) \__ \   which supports both
#  \___/ \____|_| \_\ |_| \_|\__,_|\__|\__,_|_|  \__,_|_| |_|   |_| |_|\___/ \__\___/|___/   training & classification
#
# (c) Shahar Gino, Feb-2017, sgino209@gmail.com

from main_bnb import main_bnb
from main_knn import main_knn
from main_rfc import main_rfc
from main_svm import main_svm
from common import Struct
from time import time
import getopt
import sys

# ---------------------------------------------------------------------------------------------------------------
def usage():
    print 'ocr_np.py -t [train_img_base] -x [test_img_base] -p [pickle_path] -c [cm_dump_path]'
    print 'Optional flags: --cv_engine, --training_en, --no_classification, --no_pickle, --plot_en, --knn_study_en'

# ---------------------------------------------------------------------------------------------------------------
def main(argv):

    # Default parameters:
    cv_engine = "SVM"  # BNB / KNN / RFC / SVM / VJ

    args = Struct(
        training_en = True,       # Training enabling (model fit)
        classification_en = True,  # Classification enabling (predict)

        svm_digits_only = False,   # SVM model fits over sklearn digits database, instead of Charls74 database

        pickle_en    = True,   # Save the trained classifier as pickle files (in args.pickle_path)
        print_cm_en  = True,   # Print Confusion-Matrix
        knn_study_en = False,  # KNN study, for obtaining best K parameter (for CV_ENGINE = "KNN" only)
        plot_en      = True,   # Plots enabling (sample), during classification

        train_img_base = "/Users/shahargino/Documents/ImageProcessing/Zenith_Watch/databases/Charls74/train",
        test_img_base  = "/Users/shahargino/Documents/ImageProcessing/Zenith_Watch/databases/Charls74/test",
        pickle_path    = "/Users/shahargino/Documents/ImageProcessing/Zenith_Watch/code/trained_models",
        cm_dump_path   = "/Users/shahargino/Documents/ImageProcessing/Zenith_Watch/code/confusion_matrices",

        img_size = 50
        )

    # -- .. -- .. -- .. -- .. -- .. -- .. -- .. -- .. -- .. -- .. -- .. -- .. -- .. -- .. -- .. -- ..
    # User-Arguments parameters (overrides Defaults):
    try:
        opts, user_args = getopt.getopt(argv, "ht:x:p:c:",
                                        ["cv_engine=", "training_en", "no_classification",
                                         "no_pickle", "plot_en", "knn_study_en"])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for opt, user_arg in opts:
        if opt == '-h':
            usage()
            sys.exit()
        elif opt in "-t":
            args.train_img_base = user_arg
        elif opt in "-x":
            args.test_img_base = user_arg
        elif opt in "-p":
            args.pickle_path = user_arg
        elif opt in "-c":
            args.cm_dump_path = user_arg
        elif opt in "--cv_engine":
            cv_engine = user_arg
        elif opt in "--training_en":
            args.training_en = True
        elif opt in "--no_classification":
            args.classification_en = False
        elif opt in "--no_pickle":
            args.pickle_en = False
        elif opt in "--plot_en":
            args.plot_en = True
        elif opt in "--knn_study_en":
            args.knn_study_en = True

    # -- .. -- .. -- .. -- .. -- .. -- .. -- .. -- .. -- .. -- .. -- .. -- .. -- .. -- .. -- .. -- ..
    # Call for sub-main engines:
    predictions = {}

    # Bernoulli Naive Bayes:
    if cv_engine == "BNB":
        predictions = main_bnb(args)

    # K-Nearest Neighbors (k=5):
    elif cv_engine == "KNN":
        predictions = main_knn(args)

    # Random Forest Classifier:
    elif cv_engine == "RFC":
        predictions = main_rfc(args)

    # Support Vector Machines:
    elif cv_engine == "SVM":
        predictions = main_svm(args)

    # Viola Jones:
    elif cv_engine == "VJ":
        print('Viola-Jones is under construction...')
        pass

    # -- .. -- .. -- .. -- .. -- .. -- .. -- .. -- .. -- .. -- .. -- .. -- .. -- .. -- .. -- .. -- ..
    # Print results:
    for image_file, prediction in predictions.iteritems():
        print("%s Prediction: %s --> %s" % (cv_engine, image_file, prediction))

# ---------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    t0 = time()
    print 'Start'

    main(sys.argv[1:])

    t1 = time()
    t_elapsed_sec = t1 - t0
    print('Done! (%.2f sec)' % t_elapsed_sec)
