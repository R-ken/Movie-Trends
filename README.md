# 353Project

Everything was done originally with github and everything was transfered here for submission.

https://github.com/R-ken/353Project

# What has been done so far

1. See if audience average and critic average correlate with one another. Calculated the correlation coefficient and printed the value to see how they relate to one another.

2. Split movies so one set contains movies from 2000 and before and the other is 2000 and after. Take the audience average of them both and do a mann whitney test to see if people like the older movies more or the newer ones

3. Predicting audience average based on audience percent, critic average, and critic percent

4. Predicting whether a movie will have a good (>=80%) critic rating based on the cast 


# How to Run

The Project.py file can be run on its own with no required arguments. The file will calculate and print the p-values of different tests.

To build the model for critic rating based on cast, the predict_success_notebook.ipynb should be run. 

# Required Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn import preprocessing
