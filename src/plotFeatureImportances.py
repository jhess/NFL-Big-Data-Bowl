import matplotlib.pyplot as plt
import numpy as np

def plotFeatureImportances(mdl, X_train):
    # plot feature importances using a forests of trees regression model
    importances = mdl.feature_importances_
    print(importances)
    std = np.std([tree.feature_importances_ for tree in mdl.estimators_],
                axis=0)
    indices = np.argsort(importances)[::-1][:20]
    print(indices)

    plt.figure(figsize=(14,10))
    plt.title("NFL Play Feature Importances")
    plt.bar(range(len(indices)), importances[indices],
        color="b", yerr=std[indices], align="center")
    plt.xticks(range(len(indices)), X_train.columns[indices], rotation=90)
    plt.xlim([-1, len(indices)])
    plt.ylabel("Percentile weight")
    plt.xlabel("Play Feature")
    plt.savefig('feat_importances.png')
    plt.show()