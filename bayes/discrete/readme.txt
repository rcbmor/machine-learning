
This is a simple Naive Bayes binary classifier for categorized attributes.
Binary meas it supports only 2 output classes.

#-+-
 $ python binbayes.py

usage: binbayes.py [-h] [--test] [--analysis {holdout,kfolds}]
                   [--dataset DATASET] [--names NAMES] [--positive POSITIVE]

Naive Beyes Classifier - Ricardo Moreira and Ian Rosadas

optional arguments:
  -h, --help            show this help message and exit
  --test
  --analysis {holdout,kfolds}
                        chose performance analysis
  --dataset DATASET     the csv file of the dataset
  --names NAMES         a csv file with attribute names
  --positive POSITIVE   value for positive class

#-+-

Dataset for testing/debug:

# ./datasets/playgolf.names
outlook,temperature,humidity,windy,playgolfrainy,hot,high,false,no

# datasets/playgolf.data
rainy,hot,high,true,no
overcast,hot,high,false,yes
sunny,mild,high,false,yes
sunny,cool,normal,false,yes
sunny,cool,normal,true,no
overcast,cool,normal,true,yes
rainy,mild,high,false,no
rainy,cool,normal,false,yes
sunny,mild,normal,false,yes

To run with sample:

#-+-
 $ python binbayes.py --analysis holdout --dataset datasets/playgolf.data --names ./datasets/playgolf.names --positive yes

Dataset: train = 9 , test = 5

Confusion Matrix
  ['T', 'F']
T  (3, 1)
F  (0, 1)

Performance: 
F-Score     = 0.7742
Precision   = 0.75 
Revocation  = 0.75 
Accuracy    = 0.8  

#-+-

To run with UCI database:

#-+-
 $ python binbayes.py --analysis kfolds --dataset datasets/tic-tac-toe.data --names ./datasets/tic-tac-toe.names --positive positive

Dataset: 1 , train = 319 , test = 319
Dataset: 2 , train = 319 , test = 319
Dataset: 3 , train = 319 , test = 319
Confusion Matrix
  ['T', 'F']
T  (536, 158)
F  (174, 89)

Performance: 
F-Score     = 0.7859
Precision   = 0.7723
Revocation  = 0.8576
Accuracy    = 0.7252

#-+-
