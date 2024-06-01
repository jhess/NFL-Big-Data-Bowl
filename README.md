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
GameId - a unique game identifier <br />
PlayId - a unique play identifier <br />
Team - home or away <br />
X - player position along the long axis of the field. See figure below. <br />
Y - player position along the short axis of the field. See figure below. <br />
S - speed in yards/second <br />
A - acceleration in yards/second^2 <br />
Dis - distance traveled from prior time point, in yards <br />
Orientation - orientation of player (deg) <br />
Dir - angle of player motion (deg) <br />
NflId - a unique identifier of the player <br />
DisplayName - player's name <br />
JerseyNumber - jersey number <br />
Season - year of the season <br />
YardLine - theyard line of the line of scrimmage <br />
Quarter - game quarter (1-5, 5 == overtime) <br />
GameClock - time on the game clock <br />
PossessionTeam - team with possession <br />
Down - the down (1-4) <br />
Distance - yards needed for a first down <br />
FieldPosition - which side of the field the play is happening on <br />
HomeScoreBeforePlay - home team score before play started <br />
VisitorScoreBeforePlay - visitor team score before play started <br />
NflIdRusher - the NflId of the rushing player <br />
OffenseFormation - offense formation <br />
OffensePersonnel - offensive team positional grouping <br />
DefendersInTheBox - number of defenders lined up near the line of scrimmage, spanning the width
of the offensive line <br />
DefensePersonnel - defensive team positional grouping <br />
PlayDirection - direction the play is headed <br />
TimeHandoff - UTC time of the handoff <br />
TimeSnap - UTC time of the snap <br />
**Yards - the yardage gained on the play (RESPONSE VARIABLE)** <br />
PlayerHeight - player height (ft-in) <br />
PlayerWeight - player weight (lbs) <br />
PlayerBirthDate - birthdate (mm/dd/yyyy) <br />
PlayerCollegeName - where the player attended college <br />
Position - the player's position (the specific role on the field that they typically play) <br />
HomeTeamAbbr - home team abbreviation <br />
VisitorTeamAbbr - visitor team abbreviation <br />
Week - week into the season <br />
Stadium - stadium where the game is being played <br />
Location - city where the game is being played <br />
StadiumType - description of the stadium environment <br />
Turf - description of the field surface <br />
GameWeather - description of the game weather <br />
Temperature - temperature (deg F) <br />
Humidity - humidity <br />
WindSpeed - wind speed in miles/hour <br />
WindDirection - wind direction <br />

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

<img src="https://github.com/jhess/NFL-Big-Data-Bowl/assets/1844404/568a71ff-a545-4c03-96c4-4d7638110128" />
<div style="text-align:center">
  Figure 1: Distribution of Yardage in Training Data 
</div>

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
`python naive_bayes.py`

# Naive Bayes Model
To generate Naive Bayes results, first ensure that preprocess.py is pointing to the correct directory if the default is not the location of your training data. Then, run:
`python naive_bayes.py`

Exit figures to continue program
