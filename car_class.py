import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, RocCurveDisplay, plot_roc_curve, ConfusionMatrixDisplay
from sklearn.preprocessing import RobustScaler
import re
import matplotlib.pyplot as plt

def str_to_num(frame):
    # This function creates a label for each uniquee value for the desired feature
    legend = {}
    count = 0
    for text in frame:
        if text not in legend:
            legend[text] = count
            count += 1

    return legend

def feat_sel(xdata, ydata):
    # this function uses the select k best algorithm to determine the most important features
    model = SelectKBest(score_func=f_classif, k="all")
    model.fit(xdata,ydata)
    return model.scores_

def data_process(filename):
    # this function processes the data to create the labels for each of the features. 
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

    labeled_price = []
    # labeled_price = [item >= 14500 for item in data["price"].to_numpy()]
    for item in data["price"].to_numpy():
        if item >= 14500:
            labeled_price += [0]
        else:
            labeled_price += [1]
    data["labeled_price"] = labeled_price

    # labeled_price = [item >= 14500 for item in data["price"].to_numpy()]
    # data["labeled_price"] = labeled_price

    scale_mileage = data[["mileage"]]
    transformer = RobustScaler().fit_transform(scale_mileage)
    data["mileage"] = transformer
    # print(data.describe())
    return data

def select_classifier(xdata, ydata, neighbors, outfile):

    xtrain, xtest, ytrain, ytest = train_test_split(xdata, ydata, random_state=0, test_size=0.3)

    model = KNeighborsClassifier(n_neighbors=neighbors, weights="distance")
    model.fit(xtrain, ytrain)

    pred = model.predict(xtest)

    # model = KNeighborsClassifier(n_neighbors=neighbors, weights="distance") 
    # model.fit(xdata, ydata)

    # pred = model.predict(xdata)

    cv_scores = cross_val_score(model, xtest, ytest, cv=5, scoring="accuracy")
    acc_score = accuracy_score(ytest, pred)
    output = f"cv_scores mean for {neighbors} neighbors: {np.mean(cv_scores)}. Test accuracy is {acc_score}\n"
    # output2 = f"{acc_score}\n"
    output_confusion = confusion_matrix(ytest, pred)
    incorrect = output_confusion[0][1] + output_confusion[1][0]
    correct = output_confusion[0][0] + output_confusion[1][1]
    error = incorrect / (incorrect + correct)

    # RocCurveDisplay.from_predictions(ydata, pred)
    with open(outfile, "a") as f:
        f.writelines(f"{error}\n")
        f.close()
    # return[pred, ydata]
    

def price_predictor(xtrain, xtest, ytrain, ytest, neighbors, outfile):

    model = KNeighborsClassifier(n_neighbors=neighbors, weights="distance")
    model.fit(xtrain,ytrain)
    pred = model.predict(xtest)
    score = accuracy_score(ytest, pred)

    output_confusion = confusion_matrix(ytest, pred)
    incorrect = output_confusion[0][1] + output_confusion[1][0]
    correct = output_confusion[0][0] + output_confusion[1][1]
    error = incorrect / (incorrect + correct)

    true_positive = output_confusion[0][0]
    false_positive = output_confusion[1][0]
    with open(outfile, "a") as f:
        f.writelines(f"{error}\n")
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
        if item != np.min(accuracies):
            index += 1
        else:
            break
    return [index, accuracies]

def car_classify(train_filename, test_filename, train_acc_file, test_acc_file):
    
    data = data_process(train_filename)
    xdata_train = data.drop(["price", "labeled_price"], axis=1)
    ydata_train = data["labeled_price"]

    features = feat_sel(xdata_train, ydata_train)
    print(f"Higher is better \nYear: {features[0]}\nMileage: {features[1]}\nCity: {features[2]}\nState: {features[3]}\
    \nMake: {features[4]}\nModel: {features[5]}")

    if test_filename == "":
        for item in range(1,101):
            sig_xdata = xdata_train.drop(["city", "state"], axis=1)
            sig_feat = select_classifier(sig_xdata, ydata_train, item, outfile="training_error_rate2_by_k.txt")
            print(sig_feat)

    train_num_neighbors = classifier_metrics(train_acc_file)

    if test_filename != "":

        data_test = data_process(test_filename)
        xdata_train = xdata_train.drop(["city", "state"], axis=1)
        xdata_test = data_test.drop(["price", "labeled_price", "city", "state"], axis=1)
        ydata_test = data_test["labeled_price"]
        car_pred = price_predictor(xdata_train, xdata_test, ydata_train, ydata_test, neighbors=train_num_neighbors[0], outfile="test_error_rate_by_k.txt")
        print(f"The accuracy of the model is {car_pred}")

        if input("Q2? ") == "yes":
            data_test = data_process(test_filename)
            # xdata_train = xdata_train.drop(["city", "state"], axis=1)
            xdata_test = data_test.drop(["price", "labeled_price", "city", "state"], axis=1)
            ydata_test = data_test["labeled_price"]
            verify = list(range(1,101))
            for item in verify:
                car_pred = price_predictor(xdata_train, xdata_test, ydata_train, ydata_test, neighbors=item, outfile="test_error_rate2_by_k.txt")


# car_classify("pred_comp_3_and_4_training_large_W2023_v1.csv", test_filename="pred_comp_3_test_data_W2023_v1.csv",train_acc_file="training_error_rate2_by_k.txt", test_acc_file='')

def Error_by_flex(train_filename, test_filename):
    k = list(range(1,101))
    flex = [1/item for item in k]
    flex.sort()
    test = classifier_metrics(test_filename)
    train = classifier_metrics(train_filename)
    f, (ax1, ax2) = plt.subplots(2,1, sharex=True)
    ax1.scatter(x=flex, y=train[1])
    ax1.set_title("Error Rate By Flexibility")
    ax1.set_ylabel("Training Error Rate")
    ax2.scatter(x=flex, y=test[1])
    ax2.set_ylabel("Testing Error Rate")
    ax2.set_xlabel("1/K")
    plt.show()

# Error_By_Flex("training_error_rate2_by_k.txt", "test_error_rate2_by_k.txt")

def predictor_w_thresh(train_filename, test_filename):

    data_train = data_process(train_filename)
    xdata_train = data_train.drop(["price", "labeled_price", "city", "state"], axis=1)
    ydata_train = data_train["labeled_price"]

    train_xtrn, train_xtst, train_ytrn, train_ytst = train_test_split(xdata_train, ydata_train, random_state=0, shuffle=True, test_size=0.3)

    data_test = data_process(test_filename)
    xdata_test = data_test.drop(["price", "labeled_price", "city", "state"], axis=1)
    ydata_test = data_test["labeled_price"]

    model_training = KNeighborsClassifier(n_neighbors=67, weights="distance")
    model_training.fit(train_xtrn, train_ytrn)
    k67_train_pred =model_training.predict(train_xtst)

    training_roc = plot_roc_curve(estimator= model_training, X=train_xtst, y=train_ytst)
    plt.title("ROC for KNN Classifier Training Data")
    plt.show()

    training_confusion_matrix = confusion_matrix(train_ytst, k67_train_pred)
    training_cf_disp = ConfusionMatrixDisplay(confusion_matrix=training_confusion_matrix)
    training_cf_disp.plot()
    plt.title("Training Confusion Matrix")
    plt.show()

    model_testing = KNeighborsClassifier(n_neighbors=4, weights="distance")
    model_testing.fit(xdata_train, ydata_train)
    k4_test_pred =model_training.predict(xdata_test)

    testing_confusion_matrix = confusion_matrix(ydata_test, k4_test_pred)
    testing_cf_disp = ConfusionMatrixDisplay(confusion_matrix=testing_confusion_matrix)
    testing_cf_disp.plot()
    plt.title("Testing Confusion Matrix")
    plt.show()

    training_predictions = []
    testing_predictions = []
    for level in np.arange(0.1,1.0,0.1):

        train_pred = (model_training.predict_proba(train_xtst)[:,1] >= level)
        training_confusion = confusion_matrix(train_ytst,train_pred)
        training_incorrect = training_confusion[0][1] + training_confusion[1][0]
        training_correct = training_confusion[0][0] + training_confusion[1][1]
        training_error = training_incorrect / (training_incorrect + training_correct)

        test_pred = (model_testing.predict_proba(xdata_test)[:,1] >= level)
        testing_confusion = confusion_matrix(ydata_test, test_pred)
        testing_incorrect = testing_confusion[0][1] + testing_confusion[1][0]
        testing_correct = testing_confusion[0][0] + testing_confusion[1][1]
        testing_error = testing_incorrect / (testing_incorrect + testing_correct)

        training_predictions += [training_error]
        testing_predictions += [testing_error]
        with open("thresh_output.txt", "a") as f:
            f.writelines(f"{training_error}\n{testing_error}\n")
            f.close()

    f, (ax1, ax2) = plt.subplots(2,1,sharex=True)
    ax1.scatter(np.arange(0.1,1.0,0.1), training_predictions)
    ax1.set_title("Error By Classificaation Threshold")
    ax1.set_ylabel("Training Error Rate")
    ax2.scatter(np.arange(0.1,1.0,0.1), testing_predictions)
    ax2.set_ylabel("Testing Error Rate")
    ax2.set_xlabel("Classification Threshold")
    plt.show()

predictor_w_thresh("pred_comp_3_and_4_training_large_W2023_v1.csv", test_filename="pred_comp_3_test_data_W2023_v1.csv", )