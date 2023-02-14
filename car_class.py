import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.feature_selection import f_classif
from sklearn.neighbors import KNeighborsClassifier

def str_to_num(frame):
    legend = {}
    count = 0
    for text in frame:
        if text not in legend:
            legend[text] = count
            count += 1

    return legend

def car_classify(filename):
    data = pd.read_csv(filename)

    lis_city = data["city"].to_numpy()
    legend = str_to_num(lis_city)
    labeled_city = pd.Series(lis_city).map(legend)
    data["city"] = labeled_city

    lis_state = data["state"].to_numpy()
    legend = str_to_num(lis_state)
    labeled_state = pd.Series(lis_state).map(legend)
    data["state"] = labeled_state

    lis_make = data["make"].to_numpy()
    legend = str_to_num(lis_make)
    labeled_make = pd.Series(lis_make).map(legend)
    data["make"] = labeled_make

    lis_model = data["model"].to_numpy()
    legend = str_to_num(lis_model)
    labeled_model = pd.Series(lis_model).map(legend)
    data["model"] = labeled_model

    labeled_price = [item >= 14500 for item in data["price"].to_numpy()]
    data["labeled_price"] = labeled_price

    xdata = data.drop(["price", "labeled_price"], axis=1)
    ydata = data["labeled_price"]

    xtrain, xtest, ytrain, ytest = train_test_split(xdata, ydata, random_state=0, test_size=0.3)
    print(ydata.head())


car_classify("pred_comp_3_and_4_training_small_W2023_v1.csv")

# test = {"test": 0, "call": 12}
# test["test"] += 1
# test["test"] += 9
# test["two"] = 3
# print(test)
