import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.feature_selection import f_classif
from sklearn.neighbors import KNeighborsClassifier


def car_classify(filename):
    data = pd.read_csv(filename)
    print(data.describe())


car_classify("pred_comp_3_and_4_training_small_W2023_v1.csv")
