# (c) Shahar Gino, Feb-2017, sgino209@gmail.com

from common import nudge_dataset
from common import import_and_clean_data
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split


def knn_study(k_range, train_img_base, img_size):
    """
    Study for obtaining best K for the KNN classifier
    """
    x, y = import_and_clean_data(train_img_base + '/../trainLabels.csv', img_size, train_img_base)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    # Nnudge dataset (improves performance):
    x_train, y_train = nudge_dataset(x_train, y_train, img_size)

    for k in k_range:

        neighbors = KNeighborsClassifier(n_neighbors=k)
        neighbors.fit(x_train, y_train)
        acc_knn = accuracy_score(y_test, neighbors.predict(x_test))
        print('accurate_knn k=%d --> %.4f' % (k, acc_knn))
