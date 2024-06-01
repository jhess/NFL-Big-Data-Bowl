# NFL Big Data Bowl: Predicting Yard Gain

# Preprocessing
Set the datasource in preprocess.py script by modifying the variable,  train_file_path = '../../nfl-big-data-bowl-2020/train.csv'

# PCA plots
PCA plots: pca explained variance is plotted against number of components and also
pca heatmap shows the correlation of features with PCA components. 11 components are
selected as 50% of the variance is explained.

To generate the plots run:
python naive_bayes.py

# Naive Bayes Model
To generate Naive Bayes results, first ensure that preprocess.py is pointing to the correct directory if the default is not the location of your training data. Then, run:
python naive_bayes.py

Exit figures to continue program
