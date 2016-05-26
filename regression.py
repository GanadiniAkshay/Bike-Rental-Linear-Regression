import csv
import random
import numpy as np

# Load the data from CSV files and extract and return the features and labels
def load_data(file_name):
    with open(file_name,'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        feature_lists = []
        label_lists = []
        for row in spamreader:
            label_lists.append(row[-1])
            row = row[2:-3]
            feature_lists.append(row)
        features = np.matrix(np.array(feature_lists,dtype=float))
        labels = np.matrix(np.array(label_lists,dtype=float)).transpose()
        return features,labels

# Get a vector of thetas with random values associated for each theta
# The thetas correspond to the following features
# [season(1:spring,2:summer,3:fall,4:winter);year(0:2011,1:2012);month(1 to 12);hr(0 to 23);holiday(holiday or not);weekday;
#  weathersit(1,2,3,4);temp(celcius);atemp(feels like temp);humidity;windspeed]
def get_thetas():
    return np.matrix(np.array([random.randrange(1,10,1) for x in range(11)]),dtype=float).transpose()


# Calculate the hypothesis matrix
def get_hypothesis(features,labels,thetas):
    return features * thetas

# Calculate the cost function value for a given hypothesis
def cost_function(features,labels,thetas):
    hypothesis = get_hypothesis(features,labels,thetas)
    cost = hypothesis - labels
    cost = np.square(cost)
    cost = np.sum(cost)/(2*len(features))
    return cost

features_day,labels_day = load_data('day.csv')
thetas = get_thetas()
cost = cost_function(features_day,labels_day,thetas)
print cost
