from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler as stdscale
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KernelDensity

from preprocess import get_preprocessed_data
from plot_output import plot_pca_contribution
from plotFeatureImportances import plotFeatureImportances

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    """
    Uses preprocessed data to perform Machine Learning
    :return:
    """
    df = get_preprocessed_data()
    num_rows = df.shape[0]
    num_cols = df.shape[1]
    print("Number of Rows: {0:}".format(num_rows))
    print("Number of Colummns: {0:}".format(num_cols))

    # split training and testing data
    split = 0.80
    training_data = df.iloc[:int(num_rows * split), :]
    testing_data = df.iloc[int(num_rows * split):, :]

    y_train = training_data.Yards
    y_test = testing_data.Yards

    np.savetxt("y.csv", y_test, delimiter=",")

    training_data.drop('Yards', axis=1, inplace=True)
    testing_data.drop('Yards', axis=1, inplace=True)

    # scale data
    scaler = preprocessing.StandardScaler()
    training_data = pd.DataFrame(scaler.fit_transform(training_data), columns=training_data.columns)
    #print(training_data)
    testing_data = pd.DataFrame(scaler.fit_transform(testing_data), columns=testing_data.columns)
    print(testing_data)

    # perform principal component analysis
    pca = PCA()
    X = training_data[['GameId', 'PlayId', 'Team', 'X', 'Y', 'S', 'A', 'Dis', 'Orientation',
                           'Dir', 'NflId', 'DisplayName', 'JerseyNumber', 'Season', 'YardLine',
                           'Quarter', 'GameClock', 'PossessionTeam', 'Down', 'Distance',
                           'FieldPosition', 'HomeScoreBeforePlay', 'VisitorScoreBeforePlay',
                           'NflIdRusher', 'OffenseFormation', 'OffensePersonnel',
                           'DefendersInTheBox', 'DefensePersonnel', 'PlayDirection', 'TimeHandoff',
                           'TimeSnap', 'PlayerHeight', 'PlayerWeight', 'PlayerBirthDate',
                           'PlayerCollegeName', 'Position', 'HomeTeamAbbr', 'VisitorTeamAbbr',
                           'Week', 'Stadium', 'Location', 'StadiumType', 'Turf', 'GameWeather',
                           'Temperature', 'Humidity', 'WindSpeed', 'WindDirection']]
    X2 = testing_data[['GameId', 'PlayId', 'Team', 'X', 'Y', 'S', 'A', 'Dis', 'Orientation',
                           'Dir', 'NflId', 'DisplayName', 'JerseyNumber', 'Season', 'YardLine',
                           'Quarter', 'GameClock', 'PossessionTeam', 'Down', 'Distance',
                           'FieldPosition', 'HomeScoreBeforePlay', 'VisitorScoreBeforePlay',
                           'NflIdRusher', 'OffenseFormation', 'OffensePersonnel',
                           'DefendersInTheBox', 'DefensePersonnel', 'PlayDirection', 'TimeHandoff',
                           'TimeSnap', 'PlayerHeight', 'PlayerWeight', 'PlayerBirthDate',
                           'PlayerCollegeName', 'Position', 'HomeTeamAbbr', 'VisitorTeamAbbr',
                           'Week', 'Stadium', 'Location', 'StadiumType', 'Turf', 'GameWeather',
                           'Temperature', 'Humidity', 'WindSpeed', 'WindDirection']]

    pca.fit(X)
    explained_variance = 0.5
    # get the explained variance from pca
    pca_explained_variance = pca.explained_variance_ratio_

    # describe 50% of the variance
    pca = PCA(0.50)
    pca.fit(X)
    transformed_training_data = pca.transform(X)
    transformed_testing_data = pca.transform(X2)
    print(testing_data.shape)
    print(transformed_testing_data.shape)
    # plot the pca contribution of the training data
    plot_pca_contribution(np.arange(pca_explained_variance.shape[0]), np.cumsum(pca_explained_variance),
                          explained_variance, transformed_training_data.shape[1])

    #print(transformed_training_data)

    training_data.drop(['GameId','PlayId'], axis=1, inplace=True)
    testing_data.drop(['GameId','PlayId'], axis=1, inplace=True)

    # normalize the labels, y, from binary yardage vectors into log-likelihood percentiles
    ylog_train = np.where(y_train>=0, np.log(1+np.abs(y_train)), -np.log(1+np.abs(y_train)))

    np.savetxt("foo.csv", training_data, delimiter=",")

    # get the X features, removing the Yards column which is the predictor variable
    #features = training_data.drop('Yards', axis=1).select_dtypes(include=np.number).columns.tolist()
    X_train = training_data

    #featurestest = testing_data.drop('Yards', axis=1).select_dtypes(include=np.number).columns.tolist()
    X_test = testing_data

    # get transformed data through dimensionality reduction
    pca.fit(X_train)
    X_train_transformed = pca.transform(X_train)

    # Randomized Extra Trees Regression model
    mdl = ExtraTreesRegressor(
        n_estimators=500, n_jobs=-1, bootstrap=True, oob_score=True)

    # fit to the transformed data
    #mdl.fit(X_train_transformed, ylog_train)
    #mdl.fit(X_train, ylog_train)
    mdl.fit(X_train, y_train)

    # plot feature importances
    plotFeatureImportances(mdl, X_train)

    pred = mdl.oob_prediction_
    print("Pred: ", pred)
    percentiles = np.percentile(pred, list(range(10, 100, 20)))

    # do predictions
    predY = mdl.predict(X_test)
    np.savetxt("predY.csv", predY, delimiter=",")

    print("Predicted percentiles")
    print(predY)
  
    yPredYards = np.sum([predY > pct for pct in percentiles], axis=0)
    print(yPredYards)
    np.savetxt("yPredYards.csv", yPredYards, delimiter=",")

    numCorrect = sum(p == t for p, t in zip(yPredYards, y_test))
    print("Num correct:")
    print(numCorrect)
    numys = len(y_test)
    accuracy = (float(numCorrect/numys))*100
    print("Accuracy:")
    print(accuracy)
    
    #mdlLasso = 


if __name__ == '__main__':
    main()
