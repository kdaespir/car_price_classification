import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler
import re

def str_to_num(frame):
    legend = {}
    count = 0
    for text in frame:
        if text not in legend:
            legend[text] = count
            count += 1

    return legend

def feat_sel(xdata, ydata):

    model = SelectKBest(score_func=f_classif, k="all")
    model.fit(xdata,ydata)
    return model.scores_

def data_process(filename):
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

    scale_mileage = data[["mileage"]]
    transformer = RobustScaler().fit_transform(scale_mileage)
    data["mileage"] = transformer
    print(data.describe())
    return data

def select_classifier(xdata, ydata, neighbors, outfile):

    xtrain, xtest, ytrain, ytest = train_test_split(xdata, ydata, random_state=0, test_size=0.3)

    model = KNeighborsClassifier(n_neighbors=neighbors, weights="distance")
    model.fit(xtrain, ytrain)

    pred = model.predict(xtest)
    # scores = model.score(pred, ytest)

    cv_scores = cross_val_score(model, xtest, ytest, cv=5, scoring="accuracy")
    acc_score = accuracy_score(ytest, pred)
    output = f"cv_scores mean for {neighbors} neighbors: {np.mean(cv_scores)}. Test accuracy is {acc_score}\n"
    output2 = f"{acc_score}\n"
    with open(outfile, "a") as f:
        f.writelines(output2)
        f.close()
    # print(scores)
    

def price_predictor(xtrain, xtest, ytrain, ytest, neighbors, outfile):

    model = KNeighborsClassifier(n_neighbors=neighbors, weights="distance")
    model.fit(xtrain,ytrain)
    pred = model.predict(xtest)
    score = accuracy_score(ytest, pred)
    
    with open(outfile, "w") as f:
        f.writelines(f"The KNN model that was determined to be the best from the dataset was neighbors = {neighbors}\n")
        f.close()
    return score

def classifier_metrics(filename):
    
    with open(filename, "r") as f:
        data = f.readlines()
        f.close()
    
    accuracies = []
    for acc in data:
        score = acc.replace('\n', "")
        accuracies += [float(score)]

    index = 0
    for item in accuracies:
        if item != np.max(accuracies):
            index += 1
        else:
            break
    print(accuracies[index] == np.max(accuracies))
    return index

def car_classify(train_filename, test_filename, neighbors, outfile, acc_file):
    
    data = data_process(train_filename)
    xdata_train = data.drop(["price", "labeled_price"], axis=1)
    ydata_train = data["labeled_price"]

    features = feat_sel(xdata_train, ydata_train)
    print(f"Higher is better \nYear: {features[0]}\nMileage: {features[1]}\nCity: {features[2]}\nState: {features[3]}\
    \nMake: {features[4]}\nModel: {features[5]}")

    # all_feat = classifier(xdata, ydata, neighbors)
    if test_filename == "":
        sig_xdata = xdata_train.drop(["city", "state"], axis=1)
        sig_feat = select_classifier(sig_xdata, ydata_train, neighbors, outfile)

    if test_filename != "":

        data_test = data_process(test_filename)
        xdata_test = data_test.drop(["price", "labeled_price", "city", "state"], axis=1)
        ydata_test = data_test["labeled_price"]
        car_pred = price_predictor(xdata_train, xdata_test, ydata_train, ydata_test, neighbors=neighbors, outfile="Car_class_output.txt")

    

    

classifier_metrics("car_class_output_dist_wght_robscale_acc.txt")
# # car_classify("pred_comp_3_and_4_training_large_W2023_v1.csv", 16)
# for iteration in range(1,101):
#     car_classify("pred_comp_3_and_4_training_large_W2023_v1.csv", '', iteration, "car_class_output_dist_wght_robscale_acc.txt")
