from preprocess import *
from evaluation import *
from execute import *
from NaiveBayesProfessional import *
pd.options.mode.chained_assignment = None
import warnings
warnings.filterwarnings("ignore")


def plot(result1, result2, name):
    """
    Outputs a scatter plot of results from two loss-functions

    :param result1: Python list of values to be plotted on the x-axis.
    :param result2: Python list of values to be plotted on the y-axis.
    :param name: Name of the dataset to be used as title for the plot
    """
    plt.scatter(result1, result2)
    plt.title(name + " Dataset")
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.show()

def experiment(preproArr):
    """
    The main driver for the experimentation of the classifier.  Runs classifier, k-fold, cross validation, evaluations, printing and plotting of results

    :param preproArr: A Python list of Preprocessor objects.  These do not necessarily need to have any preprocessing methods called before classification
    """
    for obj in preproArr:
        #Conducts classification with 10-fold cross-validation
        n = NaiveBayesClassifier()
        e = execute(obj.df)
        results = e.crossvalidate(n.driver, 10, obj.truthColIndex)

        x = []
        y = []
        #Prints out precision and recall for each fold
        print("***" + obj.dfName + "***")
        for fold in results:
            e = Evaluation(fold[0], fold[1], np.array(obj.truthCol))
            prec = e.precision()
            rec = e.recall()
            # e.printConfusionMatrix()
            for i in range(len(prec)):
                x.append(prec[i])
                y.append(rec[i])
        print()
        print("Average precision: " + str(sum(x) / len(x)))
        print("Average recall: " + str(sum(y) / len(y)))
        print()
        #Prints scatter plots
        plot(np.array(x), np.array(y), obj.dfName)

if __name__ == '__main__':
    preProcessedArray = []
    breastCancer = pd.read_csv("Data/breast-cancer-wisconsin.csv", header=None)
    glass = pd.read_csv('Data/glass.csv', header=None)
    houseVotes = pd.read_csv('Data/house-votes-84.csv', header=None)
    iris = pd.read_csv('Data/iris.csv', header=None)
    soyBean = pd.read_csv('Data/soybean-small.csv', header=None)

    #Each dataset is put into a Preprocessor object before classification
    #A dataset is not preprocessed unless a method has been explicit called on the Preprocessor object.
    #The creation of a preprocessor object does not imply that the dataset has been modified in any way.

    breastCancerNoPre = Preprocessor(breastCancer, 10, "Breast Cancer Wisconsin")
    breastCancerNoPre.removesmissingvalues()
    preProcessedArray.append(breastCancerNoPre)

    breastCancerPre = Preprocessor(breastCancer, 10, "Breast Cancer Wisconsin - Noise Introduced")
    breastCancerPre.removesmissingvalues()
    breastCancerPre.shuffle()
    preProcessedArray.append(breastCancerPre)

    glassNoPre = Preprocessor(glass, 10, "Glass")
    preProcessedArray.append(glassNoPre)

    glassPre = Preprocessor(glass, 10, "Glass - Noise Introduced")
    glassPre.shuffle()
    preProcessedArray.append(glassPre)

    houseVotesNoPre = Preprocessor(houseVotes,0, "House Votes 84")
    preProcessedArray.append(houseVotesNoPre)

    houseVotesPre = Preprocessor(houseVotes, 0, "House Votes 84 - Noise Introduced")
    houseVotesPre.shuffle()
    preProcessedArray.append(houseVotesPre)

    #HYPER-PARAMETER! (same number of ground truth categories)
    numberOfBins = 3

    irisPre = Preprocessor(iris, 4, "Iris")
    irisPre.binning([0, 1, 2, 3], numberOfBins)
    irisPre.shuffle()
    preProcessedArray.append(irisPre)

    irisNoPre = Preprocessor(iris, 4, "Iris - Noise Introduced")
    irisPre.binning([0, 1, 2, 3], numberOfBins)
    preProcessedArray.append(irisNoPre)

    soyBeanPre = Preprocessor(soyBean, 35, "Soy bean")
    soyBeanPre.shuffle()
    preProcessedArray.append(soyBeanPre)

    soyBeanNoPre = Preprocessor(soyBean, 35, "Soy bean - Noise Introduced")
    preProcessedArray.append(soyBeanPre)

    experiment(preProcessedArray)



