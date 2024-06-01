from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.multiclass import OneVsRestClassifier

from preprocess import get_preprocessed_data
from Correlator import Correlator
from plot_output import plot_pca_contribution, plot_pcs_heatmap

import numpy as np
import pandas as pd


def main():
    """
    Uses preprocessed data to perform Machine Learning SVM
    :return:
    """
    df = get_preprocessed_data()
    num_rows = df.shape[0]
    num_cols = df.shape[1]
    print("Number of Rows: {0:}".format(num_rows))
    print("Number of Colummns: {0:}".format(num_cols))

    # digitize yards
    bins = np.array([-99, -20, -15, -10, -1.0, -0.5, 0, 0.5, 1.0, 10, 15, 20, 99])
    indices = np.digitize(df['Yards'], bins)
    print(indices)
    df = df.assign(Yards=indices)

    # split training and testing data
    split = 0.80
    input_column_list = df.columns.tolist()
    input_column_list.remove('Yards')
    training_data = df.iloc[:int(num_rows * split), :]
    testing_data = df.iloc[int(num_rows * split):, :]

    # scale data
    training_x = training_data[input_column_list]
    testing_x = testing_data[input_column_list]
    scaler = preprocessing.StandardScaler()
    training_x = pd.DataFrame(scaler.fit_transform(training_x), columns=training_x.columns)
    testing_x = pd.DataFrame(scaler.fit_transform(testing_x), columns=testing_x.columns)

    # perform principal component analysis
    pca = PCA()
    pca.fit(training_x)

    # describe 50% of the variance
    pca = PCA(0.50)
    pca.fit(training_x)
    transformed_training_x = pca.transform(training_x)
    transformed_testing_x = pca.transform(testing_x)

    # define SVM (multi class)
    clf = OneVsRestClassifier(SVC(kernel='linear'), n_jobs=-1)
    clf.fit(transformed_training_x, training_data['Yards'])
    predicted_yardage = clf.predict(transformed_testing_x)

    # confusion matrix
    c = confusion_matrix(testing_data['Yards'], predicted_yardage)
    score = accuracy_score(testing_data['Yards'], predicted_yardage)
    print(score)


if __name__ == '__main__':
    main()
