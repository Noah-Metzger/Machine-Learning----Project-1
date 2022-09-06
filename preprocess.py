import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

class Preprocessor:
    def __init__(self, df):
        """
        Constructor for Preprocessor class.  All preprocessing logic is applied to the preprocessor object.

        :param df: data table
        """
        self.df = df
        checkItems = ["?"]

    def removesmissingvalues(self):
        """
        Removes each observation that contains a missing value.
        """
        missingRows = []
        for i, row in self.df.iterrows():
            isMissing = False
            for j, value in row.items():
                for k in self.checkItems:
                    if value == k or value == math.nan:
                        isMissing = True
            if isMissing:
                missingRows.append(i)
        self.df.drop(missingRows, axis=0, inplace=True)

    def fillmean(self):
        """
        Fills each missing value with the mean value of the missing value's attribute.
        """
        for col in self.df:
            missingIndex = []
            for index, value in self.df[col].items():
                for k in self.checkItems:
                    if value == k or value == math.nan:
                        isMissing = True
            if len(missingIndex) > 0:
                newCol = self.df[col]
                newCol.drop(missingIndex)
                mean = newCol.mean()
                for i in missingIndex:
                    self.df[col][i] = mean

    def fillforward(self):
        """
        Forward fills all missing values.
        """
        for col in self.df:
            for index, value in self.df[col].items():
                for k in self.checkItems:
                    if value == k or value == math.nan:
                        if index + 1 >= len(self.df[col]):
                            self.df[col][index] = self.df[col][0]
                        else:
                            self.df[col][index] = self.df[col][index + 1]

    def fillbackward(self):
        """
        Backward fills all missing values.
        """
        for col in self.df:
            for index, value in self.df[col].items():
                for k in self.checkItems:
                    if value == k or value == math.nan:
                        if index - 1 < 0:
                            self.df[col][index] = self.df[col][len(self.df[col]) - 1]
                        else:
                            self.df[col][index] = self.df[col][index - 1]

    def labelencode(self):
        """
        Label encodes all categorical attributes.
        """
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

    def onehotencoding(self):
        """
        One hot encodes all categorical attributes
        """
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