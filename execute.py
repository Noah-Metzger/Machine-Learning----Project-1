import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None

class execute:
    def __init__(self, df):
        """
        Constructor for execute class.  Contains all logic for the classifier part of the experiment

        :param df: DataFrame of the data set
        """
        self.df = df

    def crossvalidate(self, func, nFold, truthCol):
        """
        Conducts a n fold cross-validation experiment

        :param func: The classifier function
        :param n: The fold count of the cross-validation
        :return: The results from each fold experiment
        """
        folds = self.fold(nFold)
        results = []
        for i in range(len(folds)):
            train = pd.DataFrame()
            test = pd.DataFrame()
            for j, fold in enumerate(folds):
                if j == i:
                    test = test.append(fold)
                else:
                    train = train.append(fold)

            train_response = train.iloc[:, truthCol]
            train.drop(truthCol, axis=1, inplace=True)

            test_response = test.iloc[:, truthCol]
            test.drop(truthCol, axis=1, inplace=True)

            results.append(func(train, train_response, test, test_response))

        return results

    def fold(self, n):
        """
        Splits data into n sections for a n fold experiment

        :param n: n sections of data to be split into
        :return: an numpy array of equally split sections of the data set
        """
        dfc = self.df
        return np.array_split(dfc, n)