import csv
import random
import numpy as np

thetas = np.matrix(np.array([ 1472.70183492,296.74187095,2026.30597099,27.52345753,-375.41349237,46.08374883,69.3931479,-506.20663497,-614.88739884,6272.9181255,-1003.67658687,-2232.16210837]))

# Load the data from CSV files and extract and return the features and labels
def load_data(file_path):
    global m
    with open(file_path,'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        feature_lists = []
        label_lists = []
        for row in reader:
            #label_lists.append(row[-1])
            label_lists.append(row[-1])
            feature_lists.append(row[2:-3])
        m = len(feature_lists)
        feature_lists = [[1] + feature for feature in feature_lists]
        features = np.matrix(np.array(feature_lists,dtype=float))
        labels = np.matrix(np.array(label_lists,dtype=float)).transpose()
        return features,labels

# Calculate the hypothesis matrix
def get_hypothesis(features,labels,thetas):
    return features * thetas

features,labels = load_data('./Data/Training/day.csv')

print np.subtract((get_hypothesis(features,labels,thetas.transpose())),labels)


