from preprocess import *
from numpy import random

class Evaluation:
    def __init__(self, df, pred, truth):
        """
        Constructor for Evaluation class.  All classification evaluation logic is applied to the evaluation object.

        :param df: data table
        :param pred: Predicted classification
        :param truth: Ground truth classification
        """
        self.df = df
        self.pred = pred
        self.truth = truth
        self.label = self.getlabels(truth)

    def getlabels(self, col):
        """
        Returns a non-duplicate list of the possible categorical values of a attribute.

        :param col: Categorical attribute
        :type col: Pandas Series
        :return: list of possible categories
        :rtype: list
        """

        labels = []
        for index, value in col.items():
            isDup = False
            for i in labels:
                if i == value:
                    isDup = True
            if not isDup:
                labels.append(value)
        return labels

    def getconfusionmatrix(self):
        """
        Returns a confusion matrix for the results of a classifier

        :return: a confusion matrix as a 2d list.
        """
        matrix = []
        for i in range(len(self.label)):
            matrix.append([0] * len(self.label))
        for i in range(len(self.pred)):
            matrix[self.label.index(self.pred[i])][self.label.index(self.truth[i])] += 1
        return matrix

    def truepositive(self, conmat):
        """
        Returns the count of true positives from a classification

        :param conmat: Confusion matrix
        :return: integer count of true positives
        """
        TP = []
        for i in range(len(conmat)):
            TP.append(conmat[i][i])
        return TP

    def truenegative(self, conmat):
        """
        Returns the count of true negative from a classification.

        :param conmat: Confusion matrix.
        :return: integer count of true negative.
        """
        TN = []
        for i in range(len(conmat)):
            count = 0
            for j in range(len(conmat)):
                for k in range(len(conmat)):
                    if j != i and k != i:
                        count += conmat[j][k]
            TN.append(count)
        return TN

    def falsepositive(self, conmat):
        """
        Returns the count of false positives from a classification.

        :param conmat: Confusion matrix.
        :return: integer count of false positives.
        """
        FP = []
        for i in range(len(conmat)):
            count = 0
            for j in range(len(conmat)):
                if i != j:
                    count += conmat[i][j]
            FP.append(count)
        return FP

    def falsenegative(self, conmat):
        """
        Returns the count of false negative from a classification.

        :param conmat: Confusion matrix.
        :return: integer count of false negative.
        """
        FN = []
        for i in range(len(conmat)):
            count = 0
            for j in range(len(conmat)):
                if i != j:
                    count += conmat[j][i]
            FN.append(count)
        return FN

    def precision(self):
        """
        Returns the precision values of classifier

        :return: a list of the precision of each category.
        """
        conmat = self.getconfusionmatrix(np.array(self.pred), np.array(self.truth), self.label)
        TP = self.truePositive(conmat)
        FP = self.falsePositive(conmat)
        prec = []
        for i in range(len(conmat)):
            prec.append((TP[i]) / (TP[i] + FP[i]))

        self.printResults("precision", prec)
        return prec

    def recall(self):
        """
        Returns the recall values of a classifier

        :param pred: Predicted classification from the classifier
        :param truth: Ground truth classification
        :param label: List of possible categories.
        :return: a list of the recall of each category.
        """
        conmat = self.confusionMatrix(np.array(self.pred), np.array(self.truth), self.label)
        TP = self.truePositive(conmat)
        FN = self.falseNegative(conmat)
        rec = []
        for i in range(len(conmat)):
            rec.append((TP[i]) / (TP[i] + FN[i]))

        self.printResults("recall", rec)
        return rec

    def fscore(self, b):
        """
        Returns the fScore values of a classifier.

        :param b: The importance of recall over precision.
        :param pred: Predicted classification from the classifier.
        :param truth: Ground truth classification
        :param label: List of possible categories.
        :return: a list of the fscores of each category.
        """
        prec = self.precision()
        rec = self.recall()
        f = []
        for i in range(len(self.label)):
            Fb = (1 + pow(b, 2)) * (prec[i] * rec[i]) / ((pow(b, 2) * prec[i]) + rec[i])
            f.append(Fb)

        self.printResults("f" + str(b) + "-score", f)
        return f

    def f1score(self):
        """
        Returns the f1 score values of a classifier.

        :param pred: Predicted classification from the classifier.
        :param truth: Ground truth classification
        :param label: List of possible categories.
        :return: a list of the f1 scores of each category.
        """
        return self.fScore(1)

    def printResults(self, type, results):
        out = "The " + type + " for: "
        for i, result in enumerate(results):
            out += str(self.labels[i]) + " " + str(result) + " "
        print(out)