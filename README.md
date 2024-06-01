# NFL Big Data Bowl: Predicting Yard Gain

# Description
This project aims to predict the yardage gained by an NFL team on given rushing plays through various probabilistic machine
learning models. The problem is available as a Kaggle competition
(https://www.kaggle.com/c/nfl-big-data-bowl-2020). The prediction will be in terms of a
cumulative probability distribution of the possible yardage.

# Source of Data
The data is from Kaggle, the link is at
https://www.kaggle.com/c/nfl-big-data-bowl-2020/data

# Data description
Each row in the file corresponds to a single player's involvement in a single play.
All the columns are contained in a csv file. The rows have been grouped by the PlayId column. The
following data columns are available for analysis. The response variable has been highlighted.
GameId - a unique game identifier
PlayId - a unique play identifier
Team - home or away
X - player position along the long axis of the field. See figure below.
Y - player position along the short axis of the field. See figure below.
S - speed in yards/second
A - acceleration in yards/second^2
Dis - distance traveled from prior time point, in yards
Orientation - orientation of player (deg)
Dir - angle of player motion (deg)
NflId - a unique identifier of the player
DisplayName - player's name
JerseyNumber - jersey number
Season - year of the season
YardLine - theyard line of the line of scrimmage
Quarter - game quarter (1-5, 5 == overtime)
GameClock - time on the game clock
PossessionTeam - team with possession
Down - the down (1-4)
Distance - yards needed for a first down
FieldPosition - which side of the field the play is happening on
HomeScoreBeforePlay - home team score before play started
VisitorScoreBeforePlay - visitor team score before play started
NflIdRusher - the NflId of the rushing player
OffenseFormation - offense formation
OffensePersonnel - offensive team positional grouping
DefendersInTheBox - number of defenders lined up near the line of scrimmage, spanning the width
of the offensive line
DefensePersonnel - defensive team positional grouping
PlayDirection - direction the play is headed
TimeHandoff - UTC time of the handoff
TimeSnap - UTC time of the snap
Yards - the yardage gained on the play (RESPONSE VARIABLE)
PlayerHeight - player height (ft-in)
PlayerWeight - player weight (lbs)
PlayerBirthDate - birthdate (mm/dd/yyyy)
PlayerCollegeName - where the player attended college
Position - the player's position (the specific role on the field that they typically play)
HomeTeamAbbr - home team abbreviation
VisitorTeamAbbr - visitor team abbreviation
Week - week into the season
Stadium - stadium where the game is being played
Location - city where the game is being played
StadiumType - description of the stadium environment
Turf - description of the field surface
GameWeather - description of the game weather
Temperature - temperature (deg F)
Humidity - humidity
WindSpeed - wind speed in miles/hour
WindDirection - wind direction

# Outline of approach
The response variable in our prediction model would be the yardage
achieved by a team in a play. The output is required to be interpreted in terms of a cumulative
probability distribution. That is, the cumulative probability of a team achieving a yardage less
than a given yardage value (possible yardage values are integers from -99 to 99).
The distribution of yardage values in the training data is shown in Figure 1. According to the
distribution, we intend to create category labels from -99 to 99 (In total 199 classes). We will
explore classification techniques that would classify training feature vectors to different classes
and use the prior knowledge to categorize our test data. Testing data will be prepared by an
appropriate splitting of training data (Percentage and cross validation techniques are yet to be
decided).

![yarddistribution](https://github.com/jhess/NFL-Big-Data-Bowl/assets/1844404/568a71ff-a545-4c03-96c4-4d7638110128)
Figure 1: Distribution of Yardage in Training Data

Use dimensionality reduction, specifically principal component analysis (PCA) to reduce
the total number of features, columns, into the few most important contributing features to the
overall variance of the data. We expect to explore maximum likelihood based models as it would
allow us to extract a likelihood probability vector.

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
