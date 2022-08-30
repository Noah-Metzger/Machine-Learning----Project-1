import pandas as pd
import numpy as np
import math

class Preprocessor:
    def __init__(self, df):
        self.df = df

    def removesMissingValues(self):
        checkItems = ["?"]
        missingRows = []
        for i, row in self.df.iterrows():
            isMissing = False
            for j, value in row.items():
                for k in checkItems:
                    if value == k or value == math.nan:
                        isMissing = True
            if isMissing:
                missingRows.append(i)
        self.df.drop(missingRows, axis=0, inplace=True)

    def fillMean(self):
        checkItems = ["?"]
        for col in self.df:
            missingIndex = []
            for index, value in self.df[col].items():
                for k in checkItems:
                    if value == k or value == math.nan:
                        isMissing = True
            if len(missingIndex) > 0:
                newCol = self.df[col]
                newCol.drop(missingIndex)
                mean = newCol.mean()
                for i in missingIndex:
                    self.df[col][i] = mean

    def fillForward(self):
        checkItems = ["?"]
        for col in self.df:
            for index, value in self.df[col].items():
                for k in checkItems:
                    if value == k or value == math.nan:
                        if index + 1 >= len(self.df[col]):
                            self.df[col][index] = self.df[col][0]
                        else:
                            self.df[col][index] = self.df[col][index + 1]

    def fillBackward(self):
        checkItems = ["?"]
        for col in self.df:
            for index, value in self.df[col].items():
                for k in checkItems:
                    if value == k or value == math.nan:
                        if index - 1 < 0:
                            self.df[col][index] = self.df[col][len(self.df[col]) - 1]
                        else:
                            self.df[col][index] = self.df[col][index - 1]

    def labelEncode(self):
        for col in self.df:
            if type(self.df[col][0]) == str:
                labels = []
                for index, value in self.df[col].items():
                    isDup = False
                    for i in labels:
                        if i == value:
                            isDup = True
                    if not isDup:
                        labels.append(value)
                for index, value in self.df[col].items():
                    for i in labels:
                        if i == value:
                            self.df[col][index] = labels.index(value)

    def oneHotEncoding(self):
        for col in self.df:
            if type(self.df[col][0]) == str:
                labels = []
                for index, value in self.df[col].items():
                    isDup = False
                    for i in labels:
                        if i == value:
                            isDup = True
                    if not isDup:
                        labels.append(value)
                for i in labels:
                    temp = np.zeros(self.df[col].size)
                    for index, value in self.df[col].items():
                        if i == value:
                            temp[index] = 1
                    self.df.insert(col, i, temp)
                self.df.drop(self.df.columns[[col + len(labels)]], axis=1, inplace=True)

def main():
    breastCancer = pd.read_csv(r"C:\Users\nic\Desktop\CSCI-447\project 1\Machine-Learning----Project-1\Data\breast-cancer-wisconsin.csv", header=None)
    glass = pd.read_csv('Data/glass.csv', header=None)
    houseVotes = pd.read_csv('Data/house-votes-84.csv', header=None)
    iris = pd.read_csv('Data/iris.csv', header=None)
    soyBean = pd.read_csv('Data/soybean-small.csv', header=None)
    x = Preprocessor(iris)
    x.labelEncode()

main()